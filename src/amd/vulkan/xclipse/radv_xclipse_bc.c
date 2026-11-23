/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* BC4-BC7 on Xclipse.
 *
 * The texture unit decodes BC1-BC3, but returns zeros for BC4-BC7.  Those formats are emulated
 * like ETC2 and ASTC are elsewhere: the compressed payload stays in plane 0, and every copy into
 * the image converts the copied blocks into plane 1, which views of the image sample.  Xclipse
 * samples ASTC natively, so plane 1 stays compressed where possible (radv_bc_emulation_format):
 *
 *  - BC7 and BC5_UNORM are transcoded to ASTC 4x4 (BC5 with a two-plane encoder, since its two
 *    channels are independent);
 *  - BC6H is transcoded to ASTC 4x4 HDR;
 *  - BC4_UNORM becomes the alpha half of a BC3 block, which the hardware decodes;
 *  - BC4_SNORM and BC5_SNORM, which ASTC LDR cannot represent, are decoded to R8 and R8G8 SNORM.
 */

#include "radv_xclipse.h"

#include "meta/radv_meta.h"
#include "radv_formats.h"
#include "radv_image_view.h"

#include "util/simple_mtx.h"
#include "vk_format.h"

static const uint32_t bc_decoder_spv[] = {
#include "bc_decoder_spv.h"
};

/* The same decoder with only the BC4/BC5 kinds, which compiles in milliseconds instead of a
 * fraction of a second.
 */
static const uint32_t bc45_decoder_spv[] = {
#include "bc45_decoder_spv.h"
};

static const uint32_t bc4_bc3_spv[] = {
#include "bc4_bc3_spv.h"
};

/* 3-bit weights. */
static const uint32_t bc_astc_spv[] = {
#include "bc_astc_spv.h"
};

/* 4-bit weights and a layered first-order fit, for BC7. */
static const uint32_t bc_astc_quality_spv[] = {
#include "bc_astc_quality_spv.h"
};

static const uint32_t bc6h_astc_hdr_spv[] = {
#include "bc6h_astc_hdr_spv.h"
};

enum radv_xclipse_bc_path {
   RADV_XCLIPSE_BC_DECODE,
   RADV_XCLIPSE_BC_BC4_TO_BC3,
   RADV_XCLIPSE_BC_BC5_TO_ASTC,
   RADV_XCLIPSE_BC_BC7_TO_ASTC,
   RADV_XCLIPSE_BC_BC6H_TO_ASTC,
   RADV_XCLIPSE_BC_NUM_PATHS,
};

static const struct {
   const uint32_t *code;
   size_t code_size;
   /* Specialisation constant 0: PIPELINE_MODE of bc_astc_transcoder.glsl, the encoder
    * (2 = BC7, 3 = two-plane BC5).  The other shaders have none.
    */
   uint32_t mode;
} radv_xclipse_bc_shaders[RADV_XCLIPSE_BC_NUM_PATHS] = {
   [RADV_XCLIPSE_BC_DECODE] = {bc_decoder_spv, sizeof(bc_decoder_spv), 0},
   [RADV_XCLIPSE_BC_BC4_TO_BC3] = {bc4_bc3_spv, sizeof(bc4_bc3_spv), 0},
   [RADV_XCLIPSE_BC_BC5_TO_ASTC] = {bc_astc_spv, sizeof(bc_astc_spv), 3},
   [RADV_XCLIPSE_BC_BC7_TO_ASTC] = {bc_astc_quality_spv, sizeof(bc_astc_quality_spv), 2},
   [RADV_XCLIPSE_BC_BC6H_TO_ASTC] = {bc6h_astc_hdr_spv, sizeof(bc6h_astc_hdr_spv), 0},
};

/* bc_decoder.glsl: kind of block to decode. */
enum radv_xclipse_bc_kind {
   RADV_XCLIPSE_BC4_UNORM = 0,
   RADV_XCLIPSE_BC4_SNORM = 1,
   RADV_XCLIPSE_BC5_UNORM = 2,
   RADV_XCLIPSE_BC5_SNORM = 3,
   RADV_XCLIPSE_BC7 = 4,
   RADV_XCLIPSE_BC_NUM_KINDS,
};

struct radv_xclipse_bc_state {
   simple_mtx_t mutex;

   /* Binding 0 is the output image (plane 1), binding 1 the compressed payload (plane 0). */
   VkDescriptorSetLayout ds_layout;

   /* Decode push constants: ivec3 offset, int kind, int image type, then the source in memory
    * (radv_xclipse_bc_src_push) (36 bytes).
    * Transcode push constants: ivec3 offset in blocks, uint width, height in texels, uint source
    * layout, int image type, then the source in memory (44 bytes).
    */
   VkPipelineLayout decode_layout;
   VkPipelineLayout transcode_layout;

   VkShaderModule modules[RADV_XCLIPSE_BC_NUM_PATHS];
   VkShaderModule bc45_decoder_module;
   /* [path][1]: specialised to read the blocks from memory (SRC_MEM) instead of plane 0.  The
    * decoder has one pipeline per kind of block (decode_pipelines[kind]) instead.
    */
   VkPipeline pipelines[RADV_XCLIPSE_BC_NUM_PATHS][2];
   VkPipeline decode_pipelines[RADV_XCLIPSE_BC_NUM_KINDS][2];
};

/* The end of the push constants of every shader: where the blocks of the region are in memory,
 * for the pipelines that read them from there.
 */
struct radv_xclipse_bc_src_push {
   uint32_t addr_lo, addr_hi;
   uint32_t row_pitch, slice_pitch; /* in blocks */
};

static enum radv_xclipse_bc_path
radv_xclipse_bc_path(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return RADV_XCLIPSE_BC_BC4_TO_BC3;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return RADV_XCLIPSE_BC_BC5_TO_ASTC;
   case VK_FORMAT_BC7_UNORM_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return RADV_XCLIPSE_BC_BC7_TO_ASTC;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return RADV_XCLIPSE_BC_BC6H_TO_ASTC;
   default:
      return RADV_XCLIPSE_BC_DECODE;
   }
}

/* The source layout push constant of the transcoders. */
static uint32_t
radv_xclipse_bc_src_layout(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return 3; /* SRC_BC5_UNORM */
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return 1; /* signed */
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      return 0;
   default:
      return 1; /* SRC_BC7_RAW */
   }
}

static enum radv_xclipse_bc_kind
radv_xclipse_bc_kind(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return RADV_XCLIPSE_BC4_UNORM;
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return RADV_XCLIPSE_BC4_SNORM;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return RADV_XCLIPSE_BC5_UNORM;
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return RADV_XCLIPSE_BC5_SNORM;
   default:
      return RADV_XCLIPSE_BC7;
   }
}

static VkImageViewType
radv_xclipse_bc_view_type(VkImageType image_type)
{
   switch (image_type) {
   case VK_IMAGE_TYPE_1D:
      return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
   case VK_IMAGE_TYPE_2D:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   case VK_IMAGE_TYPE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
   default:
      UNREACHABLE("bad VkImageType");
   }
}

VkResult
radv_xclipse_bc_init(struct radv_device *device)
{
   struct radv_xclipse_bc_state *bc =
      vk_zalloc(&device->meta_state.alloc, sizeof(*bc), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!bc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   simple_mtx_init(&bc->mutex, mtx_plain);
   device->meta_state.xclipse_bc = bc;
   return VK_SUCCESS;
}

void
radv_xclipse_bc_finish(struct radv_device *device)
{
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   const struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;
   const VkAllocationCallbacks *alloc = &device->meta_state.alloc;
   VkDevice _device = radv_device_to_handle(device);

   if (!bc)
      return;

   for (unsigned i = 0; i < RADV_XCLIPSE_BC_NUM_PATHS; i++) {
      disp->DestroyPipeline(_device, bc->pipelines[i][0], alloc);
      disp->DestroyPipeline(_device, bc->pipelines[i][1], alloc);
      disp->DestroyShaderModule(_device, bc->modules[i], alloc);
   }
   for (unsigned i = 0; i < RADV_XCLIPSE_BC_NUM_KINDS; i++) {
      disp->DestroyPipeline(_device, bc->decode_pipelines[i][0], alloc);
      disp->DestroyPipeline(_device, bc->decode_pipelines[i][1], alloc);
   }
   disp->DestroyShaderModule(_device, bc->bc45_decoder_module, alloc);
   disp->DestroyPipelineLayout(_device, bc->decode_layout, alloc);
   disp->DestroyPipelineLayout(_device, bc->transcode_layout, alloc);
   disp->DestroyDescriptorSetLayout(_device, bc->ds_layout, alloc);

   simple_mtx_destroy(&bc->mutex);
   vk_free(alloc, bc);
   device->meta_state.xclipse_bc = NULL;
}

/* Called with bc->mutex held. */
static VkResult
radv_xclipse_bc_init_ds_layout(struct radv_device *device, struct radv_xclipse_bc_state *bc)
{
   const struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;

   if (bc->ds_layout)
      return VK_SUCCESS;

   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = 2,
      .pBindings =
         (VkDescriptorSetLayoutBinding[]){
            {
               .binding = 0,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
               .descriptorCount = 1,
               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
               .binding = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
               .descriptorCount = 1,
               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
         },
   };

   return disp->CreateDescriptorSetLayout(radv_device_to_handle(device), &info, &device->meta_state.alloc,
                                          &bc->ds_layout);
}

/* Called with bc->mutex held. */
static VkResult
radv_xclipse_bc_create_pipeline(struct radv_device *device, struct radv_xclipse_bc_state *bc,
                                enum radv_xclipse_bc_path path, bool src_mem, uint32_t kind, VkPipeline *pipeline)
{
   const bool bc45 = path == RADV_XCLIPSE_BC_DECODE && kind <= RADV_XCLIPSE_BC5_SNORM;
   const uint32_t *code = bc45 ? bc45_decoder_spv : radv_xclipse_bc_shaders[path].code;
   const size_t code_size = bc45 ? sizeof(bc45_decoder_spv) : radv_xclipse_bc_shaders[path].code_size;
   const uint32_t push_size = (path == RADV_XCLIPSE_BC_DECODE ? 20 : 28) + sizeof(struct radv_xclipse_bc_src_push);
   VkPipelineLayout *layout = path == RADV_XCLIPSE_BC_DECODE ? &bc->decode_layout : &bc->transcode_layout;
   VkShaderModule *module = bc45 ? &bc->bc45_decoder_module : &bc->modules[path];
   const struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;
   const VkAllocationCallbacks *alloc = &device->meta_state.alloc;
   VkDevice _device = radv_device_to_handle(device);
   VkResult result;

   result = radv_xclipse_bc_init_ds_layout(device, bc);
   if (result != VK_SUCCESS)
      return result;

   if (!*layout) {
      const VkPipelineLayoutCreateInfo layout_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &bc->ds_layout,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges =
            &(VkPushConstantRange){
               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
               .size = push_size,
            },
      };
      result = disp->CreatePipelineLayout(_device, &layout_info, alloc, layout);
      if (result != VK_SUCCESS)
         return result;
   }

   if (!*module) {
      const VkShaderModuleCreateInfo module_info = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = code_size,
         .pCode = code,
      };
      result = disp->CreateShaderModule(_device, &module_info, alloc, module);
      if (result != VK_SUCCESS)
         return result;
   }

   /* Specialisation constant 3 of every shader: SRC_MEM; constant 0 is the kind of block for the
    * decoder.
    */
   const uint32_t spec[2] = {path == RADV_XCLIPSE_BC_DECODE ? kind : radv_xclipse_bc_shaders[path].mode, src_mem};
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
            .pSpecializationInfo =
               &(VkSpecializationInfo){
                  .mapEntryCount = 2,
                  .pMapEntries =
                     (VkSpecializationMapEntry[]){
                        {.constantID = 0, .offset = 0, .size = 4},
                        {.constantID = 3, .offset = 4, .size = 4},
                     },
                  .dataSize = sizeof(spec),
                  .pData = spec,
               },
         },
      .layout = *layout,
   };

   return disp->CreateComputePipelines(_device, device->meta_state.cache, 1, &pipeline_info, alloc, pipeline);
}

static VkPipeline
radv_xclipse_bc_get_pipeline(struct radv_cmd_buffer *cmd_buffer, enum radv_xclipse_bc_path path, bool src_mem,
                             enum radv_xclipse_bc_kind kind)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   VkPipeline *pipeline =
      path == RADV_XCLIPSE_BC_DECODE ? &bc->decode_pipelines[kind][src_mem] : &bc->pipelines[path][src_mem];
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&bc->mutex);
   if (!*pipeline)
      result = radv_xclipse_bc_create_pipeline(device, bc, path, src_mem, kind, pipeline);
   simple_mtx_unlock(&bc->mutex);

   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return VK_NULL_HANDLE;
   }
   return *pipeline;
}

static void
radv_xclipse_bc_bind_views(struct radv_cmd_buffer *cmd_buffer, VkPipelineLayout layout, struct radv_image_view *src,
                           struct radv_image_view *dst)
{
   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 2,
                              (VkDescriptorGetInfoEXT[]){{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                          .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                          .data.pStorageImage =
                                                             (VkDescriptorImageInfo[]){
                                                                {.imageView = radv_image_view_to_handle(dst),
                                                                 .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                                                             }},
                                                         {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                          .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                                          .data.pSampledImage = (VkDescriptorImageInfo[]){
                                                             {.imageView = radv_image_view_to_handle(src),
                                                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                                                          }}});
}

static void
radv_xclipse_bc_decode(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *src, struct radv_image_view *dst,
                       VkFormat format, const VkOffset3D *offset, const VkExtent3D *extent,
                       const struct radv_xclipse_bc_src_push *mem)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   VkPipeline pipeline =
      radv_xclipse_bc_get_pipeline(cmd_buffer, RADV_XCLIPSE_BC_DECODE, mem != NULL, radv_xclipse_bc_kind(format));

   if (pipeline == VK_NULL_HANDLE)
      return;

   radv_xclipse_bc_bind_views(cmd_buffer, bc->decode_layout, src, dst);
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   const uint32_t push_constants[9] = {
      offset->x,
      offset->y,
      offset->z,
      radv_xclipse_bc_kind(format),
      src->image->vk.image_type,
      mem ? mem->addr_lo : 0,
      mem ? mem->addr_hi : 0,
      mem ? mem->row_pitch : 0,
      mem ? mem->slice_pitch : 0,
   };
   radv_meta_push_constants(cmd_buffer, bc->decode_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                            push_constants);

   radv_unaligned_dispatch(cmd_buffer, extent->width, extent->height, extent->depth);
}

/* One invocation per block writes an ASTC 4x4 or BC3 block into plane 1, viewed as
 * R32G32B32A32_UINT (one texel per block).
 */
static void
radv_xclipse_bc_transcode(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *src,
                          struct radv_image_view *dst, VkFormat format, enum radv_xclipse_bc_path path,
                          const VkOffset3D *offset, const VkExtent3D *extent,
                          const struct radv_xclipse_bc_src_push *mem)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   VkPipeline pipeline = radv_xclipse_bc_get_pipeline(cmd_buffer, path, mem != NULL, 0);

   if (pipeline == VK_NULL_HANDLE)
      return;

   radv_xclipse_bc_bind_views(cmd_buffer, bc->transcode_layout, src, dst);
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   /* Copies into block-compressed images are block aligned, except at the image edge. */
   const uint32_t push_constants[11] = {
      offset->x / 4,
      offset->y / 4,
      offset->z,
      extent->width,
      extent->height,
      radv_xclipse_bc_src_layout(format),
      src->image->vk.image_type,
      mem ? mem->addr_lo : 0,
      mem ? mem->addr_hi : 0,
      mem ? mem->row_pitch : 0,
      mem ? mem->slice_pitch : 0,
   };
   radv_meta_push_constants(cmd_buffer, bc->transcode_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                            push_constants);

   radv_unaligned_dispatch(cmd_buffer, DIV_ROUND_UP(extent->width, 4), DIV_ROUND_UP(extent->height, 4),
                           extent->depth);
}

static void
radv_xclipse_bc_init_view(struct radv_device *device, struct radv_image_view *iview, struct radv_image *image,
                          const VkImageSubresourceLayers *subresource, VkImageAspectFlags aspect, VkFormat format,
                          VkImageUsageFlags2KHR usage)
{
   const VkImageViewUsage2CreateInfoKHR usage_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
      .usage = usage,
   };

   radv_image_view_init(iview, device,
                        &(VkImageViewCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                           .pNext = &usage_info,
                           .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                           .image = radv_image_to_handle(image),
                           .viewType = radv_xclipse_bc_view_type(image->vk.image_type),
                           .format = format,
                           .subresourceRange =
                              {
                                 .aspectMask = aspect,
                                 .baseMipLevel = subresource->mipLevel,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = subresource->baseArrayLayer +
                                               vk_image_subresource_layer_count(&image->vk, subresource),
                              },
                        },
                        NULL);
}

static void
radv_xclipse_bc_convert(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkImageLayout layout,
                        const VkImageSubresourceLayers *subresource, VkOffset3D offset, VkExtent3D extent,
                        const struct radv_xclipse_bc_src_push *mem)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const VkFormat format = image->vk.format;

   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint32_t base_slice = is_3d ? offset.z : subresource->baseArrayLayer;
   const uint32_t slice_count = is_3d ? extent.depth : vk_image_subresource_layer_count(&image->vk, subresource);

   extent = vk_image_sanitize_extent(&image->vk, extent);
   offset = vk_image_sanitize_offset(&image->vk, offset);

   /* The payload is read one block per texel: 8-byte blocks as R32G32_UINT, 16-byte blocks as
    * R32G32B32A32_UINT.  The decoded texels are written as integers.
    */
   const VkFormat load_format =
      vk_format_get_blocksize(format) == 16 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT;

   const struct radv_physical_device *pdev = radv_device_physical(device);
   const enum radv_xclipse_bc_path path = radv_xclipse_bc_path(format);
   /* Transcoded blocks are written one per texel, decoded texels as integers with the channels of
    * plane 1 (see radv_bc_emulation_format).
    */
   VkFormat store_format = VK_FORMAT_R32G32B32A32_UINT;
   if (path == RADV_XCLIPSE_BC_DECODE) {
      switch (radv_bc_emulation_format(pdev, format)) {
      case VK_FORMAT_R8_SNORM:
         store_format = VK_FORMAT_R8_UINT;
         break;
      case VK_FORMAT_R8G8_SNORM:
         store_format = VK_FORMAT_R8G8_UINT;
         break;
      default:
         store_format = VK_FORMAT_R8G8B8A8_UINT;
         break;
      }
   }
   const VkOffset3D region_offset = {offset.x, offset.y, base_slice};
   const VkExtent3D region_extent = {extent.width, extent.height, slice_count};

   struct radv_image_view src_iview, dst_iview;
   radv_xclipse_bc_init_view(device, &src_iview, image, subresource, VK_IMAGE_ASPECT_COLOR_BIT, load_format,
                             VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR);
   radv_xclipse_bc_init_view(device, &dst_iview, image, subresource, VK_IMAGE_ASPECT_PLANE_1_BIT, store_format,
                             VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);

   if (path == RADV_XCLIPSE_BC_DECODE)
      radv_xclipse_bc_decode(cmd_buffer, &src_iview, &dst_iview, format, &region_offset, &region_extent, mem);
   else
      radv_xclipse_bc_transcode(cmd_buffer, &src_iview, &dst_iview, format, path, &region_offset, &region_extent,
                                mem);

   radv_image_view_finish(&src_iview);
   radv_image_view_finish(&dst_iview);
}

/* Converts the blocks of plane 0 that a copy wrote. */
void
radv_meta_decode_bc(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkImageLayout layout,
                    const VkImageSubresourceLayers *subresource, VkOffset3D offset, VkExtent3D extent)
{
   radv_xclipse_bc_convert(cmd_buffer, image, layout, subresource, offset, extent, NULL);
}

/* Converts the blocks of a copy from memory into plane 1, reading them where the copy reads them
 * from.  Nothing waits for plane 0 then, so the copy needs no barrier, and copies that follow one
 * another overlap instead of each waiting for the previous conversion, which made uploads of many
 * small textures two to four times slower.
 */
void
radv_meta_decode_bc_from_memory(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image,
                                const VkDeviceMemoryImageCopyKHR *region)
{
   const struct vk_image_buffer_layout buf_layout = vk_image_memory_copy_layout(&image->vk, region);
   const uint64_t addr = region->addressRange.address;
   const struct radv_xclipse_bc_src_push mem = {
      .addr_lo = addr,
      .addr_hi = addr >> 32,
      .row_pitch = buf_layout.row_stride_B / buf_layout.element_size_B,
      .slice_pitch = buf_layout.image_stride_B / buf_layout.element_size_B,
   };

   radv_xclipse_bc_convert(cmd_buffer, image, region->imageLayout, &region->imageSubresource, region->imageOffset,
                           region->imageExtent, &mem);
}
