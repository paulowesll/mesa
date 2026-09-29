/*
 * Copyright 2026 Paulo Weslley
 * SPDX-License-Identifier: MIT
 *
 * BC4_UNORM -> BC3_UNORM repack for Xclipse.  The texture unit decodes BC3
 * but not BC4, and a BC4 block has exactly the layout of the BC3 alpha half,
 * so each BC4 block becomes the alpha of a BC3 block whose colour half is
 * black; the view reads it back with a (A, 0, 0, 1) swizzle.  Lossless apart
 * from the hardware's own BC3 alpha interpolation (weights in 1/64).
 *
 * Same interface as bc_astc_transcoder.glsl: one invocation per block,
 * binding 0 = plane 1 as R32G32B32A32_UINT, binding 1 = plane 0 as
 * R32G32_UINT (one texel per block).
 */
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConsts {
    ivec3 offset;      /* first block of the copy region; z = first slice */
    uint width;        /* region extent in texels */
    uint height;
    uint src_layout;   /* unused here */
    int image_type;    /* VkImageType */
} pc;

layout(set = 0, binding = 0) writeonly uniform uimage2DArray OutputImage2DArray;
layout(set = 0, binding = 0) writeonly uniform uimage3D OutputImage3D;
layout(set = 0, binding = 1) uniform utexture2DArray PayloadInput2DArray;
layout(set = 0, binding = 1) uniform utexture3D PayloadInput3D;

void main() {
    uvec3 gid = gl_GlobalInvocationID;
    if (gid.x * 4u >= pc.width || gid.y * 4u >= pc.height)
        return;

    ivec3 block = ivec3(gid) + pc.offset;
    uvec2 bc4 = pc.image_type == 2 ? texelFetch(PayloadInput3D, block, 0).xy
                                   : texelFetch(PayloadInput2DArray, block, 0).xy;
    uvec4 bc3 = uvec4(bc4, 0u, 0u);

    if (pc.image_type == 2)
        imageStore(OutputImage3D, block, bc3);
    else
        imageStore(OutputImage2DArray, block, bc3);
}
