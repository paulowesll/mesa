/*
 * Copyright © 2026 Paulo Weslley
 *
 * SPDX-License-Identifier: MIT
 */

/* BC4, BC5 and BC7 on Xclipse.
 *
 * The texture unit decodes BC1-BC3, but returns zeros for BC4-BC7.  Those formats are emulated
 * like ETC2 and ASTC are elsewhere: the compressed payload stays in plane 0, and every copy into
 * the image decodes the copied blocks into plane 1, which views of the image sample.
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

/* bc_decoder.glsl: kind of block to decode. */
enum radv_xclipse_bc_kind {
   RADV_XCLIPSE_BC4_UNORM = 0,
   RADV_XCLIPSE_BC4_SNORM = 1,
   RADV_XCLIPSE_BC5_UNORM = 2,
   RADV_XCLIPSE_BC5_SNORM = 3,
   RADV_XCLIPSE_BC7 = 4,
};

struct radv_xclipse_bc_state {
   simple_mtx_t mutex;

   /* Binding 0 is the output image (plane 1), binding 1 the compressed payload (plane 0). */
   VkDescriptorSetLayout ds_layout;

   /* Push constants: ivec3 offset, int kind, int image type. */
   VkPipelineLayout decode_layout;
   VkShaderModule decode_module;
   VkPipeline decode_pipeline;
};

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

   disp->DestroyPipeline(_device, bc->decode_pipeline, alloc);
   disp->DestroyShaderModule(_device, bc->decode_module, alloc);
   disp->DestroyPipelineLayout(_device, bc->decode_layout, alloc);
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
radv_xclipse_bc_create_pipeline(struct radv_device *device, struct radv_xclipse_bc_state *bc, const uint32_t *code,
                                size_t code_size, uint32_t push_size, VkPipelineLayout *layout, VkShaderModule *module,
                                VkPipeline *pipeline)
{
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

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
         },
      .layout = *layout,
   };

   return disp->CreateComputePipelines(_device, device->meta_state.cache, 1, &pipeline_info, alloc, pipeline);
}

static VkPipeline
radv_xclipse_bc_get_decode_pipeline(struct radv_cmd_buffer *cmd_buffer)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&bc->mutex);
   if (!bc->decode_pipeline)
      result = radv_xclipse_bc_create_pipeline(device, bc, bc_decoder_spv, sizeof(bc_decoder_spv), 20,
                                               &bc->decode_layout, &bc->decode_module, &bc->decode_pipeline);
   simple_mtx_unlock(&bc->mutex);

   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return VK_NULL_HANDLE;
   }
   return bc->decode_pipeline;
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
                       VkFormat format, const VkOffset3D *offset, const VkExtent3D *extent)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_xclipse_bc_state *bc = device->meta_state.xclipse_bc;
   VkPipeline pipeline = radv_xclipse_bc_get_decode_pipeline(cmd_buffer);

   if (pipeline == VK_NULL_HANDLE)
      return;

   radv_xclipse_bc_bind_views(cmd_buffer, bc->decode_layout, src, dst);
   radv_meta_bind_compute_pipeline(cmd_buffer, pipeline);

   const uint32_t push_constants[5] = {
      offset->x, offset->y, offset->z, radv_xclipse_bc_kind(format), src->image->vk.image_type,
   };
   radv_meta_push_constants(cmd_buffer, bc->decode_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                            push_constants);

   radv_unaligned_dispatch(cmd_buffer, extent->width, extent->height, extent->depth);
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

void
radv_meta_decode_bc(struct radv_cmd_buffer *cmd_buffer, struct radv_image *image, VkImageLayout layout,
                    const VkImageSubresourceLayers *subresource, VkOffset3D offset, VkExtent3D extent)
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

   struct radv_image_view src_iview, dst_iview;
   radv_xclipse_bc_init_view(device, &src_iview, image, subresource, VK_IMAGE_ASPECT_COLOR_BIT, load_format,
                             VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR);
   radv_xclipse_bc_init_view(device, &dst_iview, image, subresource, VK_IMAGE_ASPECT_PLANE_1_BIT,
                             VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_2_STORAGE_BIT_KHR);

   radv_xclipse_bc_decode(cmd_buffer, &src_iview, &dst_iview, format, &(VkOffset3D){offset.x, offset.y, base_slice},
                          &(VkExtent3D){extent.width, extent.height, slice_count});

   radv_image_view_finish(&src_iview);
   radv_image_view_finish(&dst_iview);
}
