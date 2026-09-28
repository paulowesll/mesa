#version 320 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp uimage2D;

/*
 * Copyright 2026 Paulo Weslley
 * SPDX-License-Identifier: MIT
 *
 * BC4/BC5/BC7 block decoders, taken from the Xclipse ICD wrapper
 * (icd/astc_compute.comp) where they fed a BC->ASTC transcoder.  Here they
 * decode straight to RGBA8 instead: this GPU samples BC1-3 natively but
 * returns zeros for BC4/BC5/BC6H/BC7, so those formats are emulated through
 * the same plane-1 path RADV already uses for ETC2 and ASTC.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

precision highp utexture2DArray;
precision highp uimage2DArray;
precision highp uimage3D;
precision highp utexture3D;

#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) writeonly uniform uimage2DArray OutputImage2DArray;
layout(set = 0, binding = 0) writeonly uniform uimage3D OutputImage3D;
layout(set = 0, binding = 1) uniform utexture2DArray PayloadInput2DArray;
layout(set = 0, binding = 1) uniform utexture3D PayloadInput3D;

layout(push_constant) uniform Registers {
    ivec3 offset;
    int vk_format;
    int vk_image_type;
} registers;

/* Mirrors the source layout enum of the wrapper shader. */
#define BC_FMT_BC4_UNORM 0
#define BC_FMT_BC4_SNORM 1
#define BC_FMT_BC5_UNORM 2
#define BC_FMT_BC5_SNORM 3
#define BC_FMT_BC7      4

int bc6h_extend_sign(int val, uint bits);
struct BC4State {
    uint p[8];
    uint w0, w1;
};

BC4State parse_bc4(uint w0, uint w1, bool isSigned) {
    BC4State st; st.w0 = w0; st.w1 = w1;
    if (!isSigned) {
        uint e0 = w0 & 255u, e1 = (w0 >> 8u) & 255u;
        st.p[0] = e0; st.p[1] = e1;
        if (e0 > e1) {
            st.p[2] = (6u*e0 + 1u*e1 + 3u)/7u; st.p[3] = (5u*e0 + 2u*e1 + 3u)/7u;
            st.p[4] = (4u*e0 + 3u*e1 + 3u)/7u; st.p[5] = (3u*e0 + 4u*e1 + 3u)/7u;
            st.p[6] = (2u*e0 + 5u*e1 + 3u)/7u; st.p[7] = (1u*e0 + 6u*e1 + 3u)/7u;
        } else {
            st.p[2] = (4u*e0 + 1u*e1 + 2u)/5u; st.p[3] = (3u*e0 + 2u*e1 + 2u)/5u;
            st.p[4] = (2u*e0 + 3u*e1 + 2u)/5u; st.p[5] = (1u*e0 + 4u*e1 + 2u)/5u;
            st.p[6] = 0u; st.p[7] = 255u;
        }
    } else {
        int e0 = bc6h_extend_sign(int(w0 & 255u), 8u);
        int e1 = bc6h_extend_sign(int((w0 >> 8u) & 255u), 8u);
        int p[8]; p[0]=e0; p[1]=e1;
        if (e0 > e1) {
            p[2] = (6*e0 + 1*e1 + 3)/7; p[3] = (5*e0 + 2*e1 + 3)/7;
            p[4] = (4*e0 + 3*e1 + 3)/7; p[5] = (3*e0 + 4*e1 + 3)/7;
            p[6] = (2*e0 + 5*e1 + 3)/7; p[7] = (1*e0 + 6*e1 + 3)/7;
        } else {
            p[2] = (4*e0 + 1*e1 + 2)/5; p[3] = (3*e0 + 2*e1 + 2)/5;
            p[4] = (2*e0 + 3*e1 + 2)/5; p[5] = (1*e0 + 4*e1 + 2)/5;
            p[6] = -128; p[7] = 127;
        }
        for(uint i=0u; i<8u; i++) st.p[i] = uint(p[i]) & 255u;
    }
    return st;
}

uint decode_bc4_pixel(BC4State st, uint i) {
    uint b = i * 3u + 16u; uint idx;
    if (b < 31u) idx = (st.w0 >> b) & 7u;
    else if (b == 31u) idx = (st.w0 >> 31u) | ((st.w1 & 3u) << 1u);
    else idx = (st.w1 >> (b - 32u)) & 7u;
    return st.p[idx];
}
uint bc_mask(uint c) { return c >= 32u ? 0xffffffffu : ((1u << c) - 1u); }
uint bc7_read(uvec4 w, inout uint p, uint c) {
    if (c == 0u) return 0u;
    uint idx = p >> 5u, bit = p & 31u;
    uint word0 = idx == 0u ? w.x : idx == 1u ? w.y : idx == 2u ? w.z : w.w;
    uint v;
    if (bit + c <= 32u) {
        v = (word0 >> bit) & bc_mask(c);
    } else {
        uint word1 = idx == 0u ? w.y : idx == 1u ? w.z : idx == 2u ? w.w : 0u;
        v = ((word0 >> bit) | (word1 << (32u - bit))) & bc_mask(c);
    }
    p += c;
    return v;
}
uint bc6h_read(uvec4 w, inout uint p, uint c) { if (c == 0u) return 0u; uint idx = p >> 5u, bit = p & 31u, word0 = idx == 0u ? w.x : idx == 1u ? w.y : idx == 2u ? w.z : w.w, v; if (bit + c <= 32u) v = (word0 >> bit) & ((1u << c) - 1u); else { uint word1 = idx == 0u ? w.y : idx == 1u ? w.z : idx == 2u ? w.w : 0u; v = (word0 >> bit) | ((word1 & ((1u << (bit + c - 32u)) - 1u)) << (32u - bit)); } p += c; return v; }
uint bc6h_read_r(uvec4 w, inout uint p, uint c) { uint v = bc6h_read(w, p, c), res = 0u; for (uint i = 0u; i < c; i++) { res = (res << 1u) | (v & 1u); v >>= 1u; } return res; }
int bc6h_extend_sign(int val, uint bits) { return (val << (32u - bits)) >> (32u - bits); }
const uint p2_packed[256] = uint[](    16842880u, 16842752u, 16842752u, 2164326400u    , 16777344u, 16777216u, 16777216u, 2164260864u    , 16843136u, 16843008u, 16843008u, 2164326656u    , 16777344u, 16842752u, 16842752u, 2164326656u    , 128u, 16777216u, 16777216u, 2164326400u    , 16842880u, 16843008u, 16843008u, 2164326657u    , 16777344u, 16842752u, 16843008u, 2164326657u    , 128u, 16777216u, 16842752u, 2164326656u    , 128u, 0u, 16777216u, 2164326400u    , 16842880u, 16843008u, 16843009u, 2164326657u    , 128u, 16777216u, 16843008u, 2164326657u    , 128u, 0u, 16777216u, 2164326656u    , 16777344u, 16843008u, 16843009u, 2164326657u    , 128u, 0u, 16843009u, 2164326657u    , 128u, 16843009u, 16843009u, 2164326657u    , 128u, 0u, 0u, 2164326657u    , 128u, 1u, 65793u, 2164326657u    , 25231744u, 16777216u, 0u, 0u    , 128u, 0u, 129u, 65793u    , 25231744u, 16842752u, 16777216u, 0u    , 25231488u, 16777216u, 0u, 0u    , 128u, 1u, 385u, 65793u    , 128u, 0u, 129u, 257u    , 16843136u, 16842752u, 16842752u, 2164260864u    , 25231488u, 16777216u, 16777216u, 0u    , 128u, 1u, 129u, 257u    , 8454528u, 65792u, 65792u, 65792u    , 25231488u, 65792u, 65792u, 257u    , 16777344u, 16843008u, 65921u, 1u    , 128u, 16843009u, 16843137u, 0u    , 25231744u, 16777216u, 1u, 65793u    , 25231488u, 16777217u, 16777217u, 257u    , 16777600u, 16777472u, 16777472u, 2164261120u    , 128u, 16843009u, 0u, 2164326657u    , 16777600u, 8454145u, 16777472u, 65537u    , 16842880u, 16842752u, 385u, 257u    , 25231488u, 257u, 16842752u, 257u    , 16777600u, 16777472u, 65665u, 65537u    , 65920u, 16777217u, 65792u, 2164260865u    , 16777600u, 65537u, 65537u, 2164261120u    , 25231744u, 16842752u, 257u, 65793u    , 16777344u, 16842752u, 385u, 1u    , 25231488u, 65536u, 256u, 257u    , 25231488u, 16842753u, 16777473u, 257u    , 8454528u, 16777217u, 16777217u, 65792u    , 16842880u, 257u, 257u, 2164326400u    , 65920u, 65792u, 16777217u, 2164260865u    , 128u, 8454400u, 65792u, 0u    , 384u, 8454401u, 256u, 0u    , 8454272u, 16843008u, 65536u, 0u    , 128u, 8454144u, 16843008u, 65536u    , 128u, 256u, 65921u, 256u    , 65920u, 257u, 16777217u, 2164326400u    , 16842880u, 65792u, 257u, 2164260865u    , 8454528u, 16842752u, 16777217u, 257u    , 25231488u, 16777217u, 257u, 65792u    , 65920u, 257u, 257u, 2164260865u    , 65920u, 16842752u, 16842752u, 2164260865u    , 16843136u, 65793u, 1u, 2164260864u    , 16777344u, 1u, 65793u, 2164326656u    , 128u, 16843009u, 16842752u, 2164326400u    , 25231488u, 16842752u, 16843009u, 0u    , 8454272u, 65536u, 65793u, 65793u    , 384u, 256u, 16843008u, 2164326656u);
const uint p3_packed[256] = uint[](    2164326528u, 16842752u, 16908800u, 2181169666u    , 2164260992u, 16842752u, 16843394u, 16908802u    , 128u, 16777218u, 16843394u, 2164326914u    , 2181169792u, 33685504u, 16842752u, 2164326656u    , 128u, 0u, 33685889u, 2181169409u    , 2164326528u, 16842752u, 33685504u, 2181169152u    , 2181169280u, 33685504u, 16843009u, 2164326657u    , 16842880u, 16842752u, 16843394u, 2164326914u    , 128u, 0u, 16843137u, 2181169666u    , 128u, 16843009u, 16843137u, 2181169666u    , 128u, 25231617u, 33686018u, 2181169666u    , 33620096u, 42008576u, 33619968u, 2181103616u    , 33620352u, 42008832u, 33620224u, 2181103872u    , 33685888u, 33718528u, 33685760u, 2181169408u    , 2164326528u, 33620224u, 33685761u, 2181169665u    , 2164326528u, 16777218u, 642u, 131586u    , 2164260992u, 16842752u, 33620224u, 2181169409u    , 2164326784u, 16842752u, 16777346u, 514u    , 128u, 33685761u, 33685889u, 2181169409u    , 2181169280u, 33685504u, 33685504u, 2164326657u    , 2164326784u, 16843008u, 33686016u, 2181169664u    , 2164260992u, 16777216u, 16908930u, 16908802u    , 128u, 25231360u, 33685760u, 2181169408u    , 128u, 257u, 8454786u, 66050u    , 2181169536u, 33718528u, 16842752u, 0u    , 33620096u, 33619968u, 33685889u, 2181169666u    , 65920u, 25297409u, 16908929u, 65792u    , 128u, 8454400u, 25297409u, 16908801u    , 33685632u, 33554689u, 33554817u, 2181169152u    , 65920u, 98560u, 33554434u, 2181169666u    , 16842880u, 33685760u, 42074368u, 2164326400u    , 128u, 2u, 16843394u, 2164392450u    , 128u, 33554432u, 33685889u, 2181169665u    , 2181169792u, 33685504u, 33619968u, 2164326400u    , 2164326528u, 33619968u, 33685504u, 2181169664u    , 131456u, 164096u, 8519936u, 131328u    , 128u, 25231617u, 42074626u, 0u    , 131456u, 16777729u, 42008706u, 131328u    , 131456u, 33619970u, 16810625u, 131328u    , 16842880u, 514u, 42074369u, 2164326400u    , 16842880u, 42074369u, 514u, 2164326400u    , 2164261248u, 16777472u, 33686018u, 2181169666u    , 128u, 0u, 16908674u, 2164392194u    , 33685632u, 33718529u, 33685504u, 2181169409u    , 2181169280u, 16842752u, 33685504u, 2164326400u    , 131712u, 25297409u, 131584u, 2164392449u    , 16777600u, 42074626u, 33686018u, 2164261120u    , 128u, 16908546u, 16908674u, 2164392194u    , 2164261248u, 16777472u, 16777472u, 2181169666u    , 2181169792u, 16843008u, 33686016u, 2164326656u    , 33554560u, 33652993u, 33554432u, 2181103873u    , 128u, 33652994u, 33620226u, 2181103874u    , 33686144u, 16875776u, 16843008u, 2181169664u    , 33554560u, 33620225u, 33620353u, 2181038080u    , 65920u, 98560u, 65792u, 2181169666u    , 128u, 0u, 42008834u, 2181103874u    , 65920u, 98560u, 33686018u, 2181169666u    , 33685632u, 16842752u, 25231360u, 2181169152u    , 33685632u, 33685761u, 33685889u, 2181169152u    , 128u, 0u, 0u, 2181136642u    , 2181038208u, 16777216u, 33554432u, 2164260864u    , 33686144u, 33686017u, 33686016u, 2181169793u    , 2164261248u, 33686018u, 33686018u, 2181169666u    , 2164326784u, 16842754u, 16777858u, 131586u);
uint get_p2(uint idx) { return (p2_packed[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u; }
uint get_p3(uint idx) { return (p3_packed[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u; }
uint bc7_w(uint a, uint b, uint w);
int bc7_w(int a, int b, uint w);
uint bc7_i2(uint a, uint b, uint i);
uint bc7_i3(uint a, uint b, uint i);
uint bc7_i4(uint a, uint b, uint i);
uvec4 bc7_exp(uvec4 e, uint b) { if (b >= 8u) return e; uint s = 8u - b; return (e << s) | (e >> (b - s)); }
uint bc7_w(uint a, uint b, uint w) { return (a * (64u - w) + b * w + 32u) >> 6u; }
int bc7_w(int a, int b, uint w) { return (a * int(64u - w) + b * int(w) + 32) >> 6; }
uint bc7_i2(uint a, uint b, uint i) { const uint w[4] = uint[](0u, 21u, 43u, 64u); return bc7_w(a, b, w[i]); }
uint bc7_i3(uint a, uint b, uint i) { const uint w[8] = uint[](0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u); return bc7_w(a, b, w[i]); }
uint bc7_i4(uint a, uint b, uint i) { const uint w[16] = uint[](0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u); return bc7_w(a, b, w[i]); }
struct BC7State {
    uint m, pid, rot, isel;
    uvec4 e[6];
    /* The first index of several BC7 modes is an anchor and has one fewer
     * bit.  Keep the decoded indices in their logical texel positions
     * instead of packing them into fixed-width words; otherwise the omitted
     * anchor bit shifts every following index and creates repeated bands or
     * isolated outliers after BC7->ASTC conversion. */
    uint idx[16];
};

BC7State parse_bc7(uvec4 w) {
    BC7State st;
    st.m = 0u; uint p = 0u; while(st.m < 8u && bc7_read(w, p, 1u) == 0u) st.m++;
    st.pid=0u; st.rot=0u; st.isel=0u;
    for(uint i=0u; i<16u; i++) st.idx[i] = 0u;
    for(uint i=0u;i<6u;i++) st.e[i] = uvec4(0u);
    if (st.m > 7u) return st;

    if (st.m == 0u) {
        st.pid = bc7_read(w, p, 4u);
        for(uint c=0u; c<3u; c++) for(uint i=0u; i<6u; i++) st.e[i][c] = bc7_read(w, p, 4u);
        for(uint i=0u; i<6u; i++) st.e[i] = bc7_exp((st.e[i]<<1u)|uvec4(bc7_read(w,p,1u)), 5u);
        for(uint i=0u; i<16u; i++) {
            uint pv=get_p3(st.pid*16u+i), b=(pv&128u)!=0u?2u:3u, val=bc7_read(w,p,b);
            st.idx[i] = val;
        }
    }
    else if (st.m == 1u) {
        st.pid = bc7_read(w, p, 6u);
        for(uint c=0u; c<3u; c++) for(uint i=0u; i<4u; i++) st.e[i][c] = bc7_read(w, p, 6u);
        uint p0=bc7_read(w,p,1u), p1=bc7_read(w,p,1u);
        for(uint i=0u; i<4u; i++) st.e[i] = bc7_exp((st.e[i]<<1u)|uvec4(i<2u?p0:p1), 7u);
        for(uint i=0u; i<16u; i++) {
            uint pv=get_p2(st.pid*16u+i), b=(pv&128u)!=0u?2u:3u, val=bc7_read(w,p,b);
            st.idx[i] = val;
        }
    }
    else if (st.m == 2u) {
        st.pid = bc7_read(w, p, 6u);
        for(uint c=0u; c<3u; c++) for(uint i=0u; i<6u; i++) st.e[i][c] = bc7_read(w, p, 5u);
        for(uint i=0u; i<6u; i++) st.e[i] = bc7_exp(st.e[i], 5u);
        for(uint i=0u; i<16u; i++) {
            uint pv=get_p3(st.pid*16u+i), b=(pv&128u)!=0u?1u:2u, val=bc7_read(w,p,b);
            st.idx[i] = val;
        }
    }
    else if (st.m == 3u) {
        st.pid = bc7_read(w, p, 6u);
        for(uint c=0u; c<3u; c++) for(uint i=0u; i<4u; i++) st.e[i][c] = bc7_read(w, p, 7u);
        for(uint i=0u; i<4u; i++) st.e[i] = bc7_exp((st.e[i]<<1u)|uvec4(bc7_read(w,p,1u)), 8u);
        for(uint i=0u; i<16u; i++) {
            uint pv=get_p2(st.pid*16u+i), b=(pv&128u)!=0u?1u:2u, val=bc7_read(w,p,b);
            st.idx[i] = val;
        }
    }
    else if (st.m == 4u) {
        st.rot = bc7_read(w, p, 2u); st.isel = bc7_read(w, p, 1u);
        for(uint c=0u; c<3u; c++) { st.e[0][c] = bc7_read(w, p, 5u); st.e[1][c] = bc7_read(w, p, 5u); }
        st.e[0].a = bc7_read(w, p, 6u); st.e[1].a = bc7_read(w, p, 6u);
        st.e[0].rgb = bc7_exp(st.e[0], 5u).rgb; st.e[1].rgb = bc7_exp(st.e[1], 5u).rgb;
        st.e[0].a = bc7_exp(st.e[0], 6u).a; st.e[1].a = bc7_exp(st.e[1], 6u).a;
        for(uint i=0u; i<16u; i++)
            st.idx[i] = bc7_read(w,p,i==0u?1u:2u);
        for(uint i=0u; i<16u; i++) {
            uint val=bc7_read(w,p,i==0u?2u:3u);
            st.idx[i] |= val << 2u;
        }
    }
    else if (st.m == 5u) {
        st.rot = bc7_read(w, p, 2u);
        for(uint c=0u; c<3u; c++) { st.e[0][c] = bc7_read(w, p, 7u); st.e[1][c] = bc7_read(w, p, 7u); }
        st.e[0].a = bc7_read(w, p, 8u); st.e[1].a = bc7_read(w, p, 8u);
        st.e[0].rgb = bc7_exp(st.e[0], 7u).rgb; st.e[1].rgb = bc7_exp(st.e[1], 7u).rgb;
        for(uint i=0u; i<16u; i++)
            st.idx[i] = bc7_read(w,p,i==0u?1u:2u);
        for(uint i=0u; i<16u; i++)
            st.idx[i] |= bc7_read(w,p,i==0u?1u:2u) << 2u;
    }
    else if (st.m == 6u) {
        for(uint c=0u; c<4u; c++) { st.e[0][c] = bc7_read(w, p, 7u); st.e[1][c] = bc7_read(w, p, 7u); }
        uint p0=bc7_read(w,p,1u), p1=bc7_read(w,p,1u);
        st.e[0] = bc7_exp((st.e[0]<<1u)|uvec4(p0), 8u); st.e[1] = bc7_exp((st.e[1]<<1u)|uvec4(p1), 8u);
        for(uint i=0u; i<16u; i++)
            st.idx[i] = bc7_read(w,p,i==0u?3u:4u);
    }
    else if (st.m == 7u) {
        st.pid = bc7_read(w, p, 6u);
        for(uint c=0u; c<3u; c++) for(uint i=0u; i<4u; i++) st.e[i][c] = bc7_read(w, p, 5u);
        for(uint i=0u; i<4u; i++) st.e[i].a = bc7_read(w, p, 5u);
        for(uint i=0u; i<4u; i++) st.e[i] = bc7_exp((st.e[i]<<1u)|uvec4(bc7_read(w,p,1u)), 6u);
        for(uint i=0u; i<16u; i++) {
            uint pv=get_p2(st.pid*16u+i), b=(pv&128u)!=0u?1u:2u, val=bc7_read(w,p,b);
            st.idx[i] = val;
        }
    }
    return st;
}

uvec4 decode_bc7_pixel(BC7State st, uint i) {
    if (st.m > 7u) return uvec4(0u);
    if (st.m == 0u) {
        uint pv=get_p3(st.pid*16u+i), ts=pv&3u, ei=ts*2u;
        uint tw=st.idx[i];
        return uvec4(bc7_i3(st.e[ei].r,st.e[ei+1u].r,tw), bc7_i3(st.e[ei].g,st.e[ei+1u].g,tw), bc7_i3(st.e[ei].b,st.e[ei+1u].b,tw), 255u);
    }
    if (st.m == 1u) {
        uint pv=get_p2(st.pid*16u+i), ts=pv&3u, ei=ts*2u;
        uint tw=st.idx[i];
        return uvec4(bc7_i3(st.e[ei].r,st.e[ei+1u].r,tw), bc7_i3(st.e[ei].g,st.e[ei+1u].g,tw), bc7_i3(st.e[ei].b,st.e[ei+1u].b,tw), 255u);
    }
    if (st.m == 2u) {
        uint pv=get_p3(st.pid*16u+i), ts=pv&3u, ei=ts*2u;
        uint tw=st.idx[i];
        return uvec4(bc7_i2(st.e[ei].r,st.e[ei+1u].r,tw), bc7_i2(st.e[ei].g,st.e[ei+1u].g,tw), bc7_i2(st.e[ei].b,st.e[ei+1u].b,tw), 255u);
    }
    if (st.m == 3u) {
        uint pv=get_p2(st.pid*16u+i), ts=pv&3u, ei=ts*2u;
        uint tw=st.idx[i];
        return uvec4(bc7_i2(st.e[ei].r,st.e[ei+1u].r,tw), bc7_i2(st.e[ei].g,st.e[ei+1u].g,tw), bc7_i2(st.e[ei].b,st.e[ei+1u].b,tw), 255u);
    }
    if (st.m == 4u) {
        uint c_v=st.idx[i] & 3u;
        uint a_v=st.idx[i] >> 2u;
        uint ic=st.isel!=0u?a_v:c_v, ia=st.isel!=0u?c_v:a_v;
        uint r=st.isel!=0u?bc7_i3(st.e[0].r,st.e[1].r,ic):bc7_i2(st.e[0].r,st.e[1].r,ic);
        uint g=st.isel!=0u?bc7_i3(st.e[0].g,st.e[1].g,ic):bc7_i2(st.e[0].g,st.e[1].g,ic);
        uint b=st.isel!=0u?bc7_i3(st.e[0].b,st.e[1].b,ic):bc7_i2(st.e[0].b,st.e[1].b,ic);
        uint a=st.isel!=0u?bc7_i2(st.e[0].a,st.e[1].a,ia):bc7_i3(st.e[0].a,st.e[1].a,ia);
        if(st.rot==1u){uint t=a;a=r;r=t;} else if(st.rot==2u){uint t=a;a=g;g=t;} else if(st.rot==3u){uint t=a;a=b;b=t;}
        return uvec4(r,g,b,a);
    }
    if (st.m == 5u) {
        uint c_v=st.idx[i] & 3u, a_v=st.idx[i] >> 2u;
        uint r=bc7_i2(st.e[0].r,st.e[1].r,c_v), g=bc7_i2(st.e[0].g,st.e[1].g,c_v), b=bc7_i2(st.e[0].b,st.e[1].b,c_v), a=bc7_i2(st.e[0].a,st.e[1].a,a_v);
        if(st.rot==1u){uint t=a;a=r;r=t;} else if(st.rot==2u){uint t=a;a=g;g=t;} else if(st.rot==3u){uint t=a;a=b;b=t;}
        return uvec4(r,g,b,a);
    }
    if (st.m == 6u) {
        uint idx=st.idx[i];
        return uvec4(bc7_i4(st.e[0].r,st.e[1].r,idx), bc7_i4(st.e[0].g,st.e[1].g,idx), bc7_i4(st.e[0].b,st.e[1].b,idx), bc7_i4(st.e[0].a,st.e[1].a,idx));
    }
    if (st.m == 7u) {
        uint pv=get_p2(st.pid*16u+i), ts=pv&3u, ei=ts*2u;
        uint tw=st.idx[i];
        return uvec4(bc7_i2(st.e[ei].r,st.e[ei+1u].r,tw), bc7_i2(st.e[ei].g,st.e[ei+1u].g,tw), bc7_i2(st.e[ei].b,st.e[ei+1u].b,tw), bc7_i2(st.e[ei].a,st.e[ei+1u].a,tw));
    }
    return uvec4(0u);
}

uvec4 decode_texel(uvec4 payload, uint texel)
{
    if (registers.vk_format == BC_FMT_BC7)
        return decode_bc7_pixel(parse_bc7(payload), texel);

    bool is_signed = registers.vk_format == BC_FMT_BC4_SNORM ||
                     registers.vk_format == BC_FMT_BC5_SNORM;
    uint r = decode_bc4_pixel(parse_bc4(payload.x, payload.y, is_signed), texel);

    if (registers.vk_format == BC_FMT_BC5_UNORM ||
        registers.vk_format == BC_FMT_BC5_SNORM) {
        uint g = decode_bc4_pixel(parse_bc4(payload.z, payload.w, is_signed), texel);
        return uvec4(r, g, 0u, 255u);
    }
    return uvec4(r, 0u, 0u, 255u);
}

void main()
{
    ivec3 coord = ivec3(gl_GlobalInvocationID) + registers.offset;
    bool is_3d = registers.vk_image_type == 2; /* VK_IMAGE_TYPE_3D */

    ivec3 block = ivec3(coord.x >> 2, coord.y >> 2, coord.z);
    uvec4 payload = is_3d ? texelFetch(PayloadInput3D, block, 0)
                          : texelFetch(PayloadInput2DArray, block, 0);

    uint texel = uint((coord.y & 3) * 4 + (coord.x & 3));
    uvec4 color = decode_texel(payload, texel);

    if (is_3d)
        imageStore(OutputImage3D, coord, color);
    else
        imageStore(OutputImage2DArray, coord, color);
}
