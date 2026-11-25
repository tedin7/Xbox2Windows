; Source-built quad for two wave64s that samples one texture with a sampler loaded from the
; sampler heap. s17 points to a descriptor table: the geometry buffer at byte 0x20 and the
; texture at byte 0x40. s16 holds the sampler heap base.
; Geometry dword 0 is a sampler heap byte offset. 0x10 and 0x20 are loaded through the register;
; any other value takes the constant offset 0x30.
.text
s_mov_b32 s10, s17
s_mov_b32 s11, 0
s_load_dwordx4 s[24:27], s[10:11], 0x20
s_load_dwordx8 s[28:35], s[10:11], 0x40
s_mov_b32 s12, s16
s_mov_b32 s13, 0
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s5, s[24:27], null
s_waitcnt lgkmcnt(0)
s_cmp_eq_u32 s5, 16
s_cbranch_scc0 check_second
s_branch register_offset
check_second:
s_cmp_eq_u32 s5, 32
s_cbranch_scc0 constant_offset
register_offset:
s_load_dwordx4 s[40:43], s[12:13], s5
s_branch sampler_ready
constant_offset:
s_load_dwordx4 s[40:43], s[12:13], 0x30
sampler_ready:
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
s_mov_b64 s[22:23], exec
v_cmpx_eq_u32_e32 0, v0
s_cbranch_execz finish_alloc
s_mov_b32 m0, 0x2004
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
finish_alloc:
s_mov_b64 exec, s[22:23]
s_barrier
s_waitcnt lgkmcnt(0)
v_cmpx_gt_u32_e32 4, v0
v_and_b32_e32 v10, 1, v0
v_and_b32_e32 v11, 2, v0
v_cmp_eq_u32_e64 s[20:21], 1, v10
v_cmp_eq_u32_e64 s[14:15], 2, v11
v_cndmask_b32_e64 v1, -1.0, 1.0, s[20:21]
v_cndmask_b32_e64 v2, -1.0, 1.0, s[14:15]
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v7, 0x3fa00000
v_mov_b32_e32 v8, 0.5
image_sample_lz v[12:15], v[7:8], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp pos0 v1, v2, v3, v4 done
exp param0 v12, v13, v14, v15
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32_e32 2, v0
v_cmp_eq_u32_e64 s[20:21], 1, v0
v_mov_b32_e32 v9, 0x200400
v_mov_b32_e32 v6, 0x300402
v_cndmask_b32_e64 v9, v9, v6, s[20:21]
exp prim v9, off, off, off done
s_endpgm
