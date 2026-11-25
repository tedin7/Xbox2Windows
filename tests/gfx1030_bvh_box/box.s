// SPDX-License-Identifier: MIT
.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"

.text
.p2align 8
.globl bvh32
.type bvh32,@function
bvh32:
  s_load_dwordx4 s[4:7], s[0:1], 0x00
  s_load_dwordx2 s[8:9], s[0:1], 0x10
  s_load_dwordx4 s[12:15], s[0:1], 0x20
  s_load_dwordx4 s[16:19], s[0:1], 0x30
  s_load_dwordx4 s[20:23], s[0:1], 0x40
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v16, s12
  v_mov_b32_e32 v17, s14
  v_mov_b32_e32 v18, s15
  v_mov_b32_e32 v19, s16
  v_mov_b32_e32 v20, s17
  v_mov_b32_e32 v21, s18
  v_mov_b32_e32 v22, s19
  v_mov_b32_e32 v23, s20
  v_mov_b32_e32 v24, s21
  v_mov_b32_e32 v25, s22
  v_mov_b32_e32 v26, s23
  image_bvh_intersect_ray v[32:35], v[16:31], s[4:7]
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v0, 0
  global_store_dwordx4 v0, v[32:35], s[8:9]
  s_waitcnt_vscnt null, 0
  s_endpgm
.size bvh32, .-bvh32
.rodata
.p2align 6
.amdhsa_kernel bvh32
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 80
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 36
  .amdhsa_next_free_sgpr 24
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.text
.p2align 8
.globl bvh64
.type bvh64,@function
bvh64:
  s_load_dwordx4 s[4:7], s[0:1], 0x00
  s_load_dwordx2 s[8:9], s[0:1], 0x10
  s_load_dwordx4 s[12:15], s[0:1], 0x20
  s_load_dwordx4 s[16:19], s[0:1], 0x30
  s_load_dwordx4 s[20:23], s[0:1], 0x40
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v16, s12
  v_mov_b32_e32 v17, s13
  v_mov_b32_e32 v18, s14
  v_mov_b32_e32 v19, s15
  v_mov_b32_e32 v20, s16
  v_mov_b32_e32 v21, s17
  v_mov_b32_e32 v22, s18
  v_mov_b32_e32 v23, s19
  v_mov_b32_e32 v24, s20
  v_mov_b32_e32 v25, s21
  v_mov_b32_e32 v26, s22
  v_mov_b32_e32 v27, s23
  image_bvh64_intersect_ray v[32:35], v[16:31], s[4:7]
  s_waitcnt vmcnt(0)
  v_mov_b32_e32 v0, 0
  global_store_dwordx4 v0, v[32:35], s[8:9]
  s_waitcnt_vscnt null, 0
  s_endpgm
.size bvh64, .-bvh64
.rodata
.p2align 6
.amdhsa_kernel bvh64
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 80
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 36
  .amdhsa_next_free_sgpr 24
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 1]
amdhsa.kernels:
  - .name: bvh32
    .symbol: bvh32.kd
    .kernarg_segment_size: 80
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 16
    .wavefront_size: 32
    .sgpr_count: 24
    .vgpr_count: 36
    .max_flat_workgroup_size: 32
  - .name: bvh64
    .symbol: bvh64.kd
    .kernarg_segment_size: 80
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 16
    .wavefront_size: 32
    .sgpr_count: 24
    .vgpr_count: 36
    .max_flat_workgroup_size: 32
...
.end_amdgpu_metadata
