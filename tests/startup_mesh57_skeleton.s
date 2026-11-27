// Source-built slot 57 mesh fixture: the group-memory count exchange and a trivial body.
// Entry: s3 bits 24..27 hold the wave index; root constants s16 = primitive count, s17 = vertex count;
// s[18:19] points at a descriptor table whose first descriptor is a raw buffer:
// 12 float4 positions at byte 0, 12 uint4 parameter words at byte 192, 12 packed index words at byte 384.
// A thread reads the slot of its low four bits, so threads past 15 stay inside the 512 bytes.
s_inst_prefetch 0x3
s_mov_b64 exec, 1
s_mov_b64 s[4:5], exec
s_and_b32 s6, s3, 0xf000000
s_cbranch_scc1 7               // to the reading waves
v_lshrrev_b32 v5, 7, v5
v_mov_b32 v6, 0
ds_write_b32 v6, v5 offset:8
s_waitcnt_depctr 0xffe3
s_barrier
s_branch 4                     // past the reading waves
s_barrier
v_mov_b32 v6, 0
ds_read_b32 v5, v6 offset:8
s_mov_b64 exec, -1
s_bfe_u32 s6, s3, 0x40018
s_lshl_b32 s7, s6, 6
v_mbcnt_lo_u32_b32 v0, -1, s7
v_mbcnt_hi_u32_b32 v0, -1, v0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s20, v5
s_load_dwordx4 s[8:11], s[18:19], 0
s_mov_b32 s21, s16
s_mov_b32 s22, s17
s_barrier
s_cmp_eq_i32 s6, 0
s_cbranch_scc0 16              // to the reading waves
s_mov_b64 exec, s[4:5]
s_cbranch_execz 7              // never taken: the saved mask is 1
v_mov_b32 v1, s21
v_mov_b32 v2, 0
v_mov_b32 v3, s22
ds_write_b32 v2, v1
ds_write_b32 v2, v3 offset:4
s_mov_b64 exec, -1
s_waitcnt_depctr 0xffe3
s_barrier
s_lshl_b32 s4, s21, 12
s_or_b32 m0, s4, s22
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
s_branch 13                    // past the reading waves
s_barrier
s_mov_b64 exec, s[4:5]
s_cbranch_execz 9              // never taken: the saved mask is 1
v_mov_b32 v2, 0
ds_read_b32 v1, v2
ds_read_b32 v3, v2 offset:4
s_waitcnt lgkmcnt(1)
v_readfirstlane_b32 s21, v1
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s22, v3
s_mov_b64 exec, -1
v_cmpx_gt_i32 s22, v0
v_and_b32 v4, 15, v0
v_lshlrev_b32 v4, 4, v4
buffer_load_dwordx4 v[16:19], v4, s[8:11], 0 offen
buffer_load_dwordx4 v[20:23], v4, s[8:11], 0 offen offset:192
s_waitcnt vmcnt(0)
exp pos0 v16, v17, v18, v19 done
exp param0 v20, v21, v22, v23
s_mov_b64 exec, -1
v_cmpx_gt_i32 s21, v0
v_and_b32 v4, 15, v0
v_lshlrev_b32 v4, 2, v4
buffer_load_dword v24, v4, s[8:11], 0 offen offset:384
s_waitcnt vmcnt(0)
exp prim v24, off, off, off done
s_endpgm
