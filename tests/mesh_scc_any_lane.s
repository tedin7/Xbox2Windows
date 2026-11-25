; Source-built two-wave64 mesh program: a group-uniform branch on "any lane active".
; s17 points to a descriptor table. Its entry at byte 0x20 names the geometry buffer.
; Geometry dword 0 is T: lanes with thread id >= T are live. Dword 1 packs the second
; triangle's indices. Each wave computes whether any of its own lanes is live and stores
; that flag at LDS byte 4 * wave. After a barrier both waves read both flags, so the
; branch outcome is the same for the whole group: the quad (4 vertices, 2 primitives) when any
; wave has a live lane, the single triangle otherwise.
.text
s_mov_b32 s10, s17
s_mov_b32 s11, 0
s_load_dwordx4 s[24:27], s[10:11], 0x20
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s15, s[24:27], null
s_buffer_load_dword s14, s[24:27], 4
s_waitcnt lgkmcnt(0)
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
v_cmp_le_u32_e32 vcc, s15, v0
s_cmp_lg_u64 vcc, 0
s_cselect_b32 s6, 1, 0
v_mov_b32_e32 v12, s6
s_lshl_b32 s9, s8, 2
v_mov_b32_e32 v13, s9
ds_write_b32 v13, v12
s_waitcnt lgkmcnt(0)
s_barrier
v_mov_b32_e32 v14, 0
ds_read_b32 v15, v14
ds_read_b32 v16, v14 offset:4
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s6, v15
v_readfirstlane_b32 s7, v16
s_or_b32 s6, s6, s7
s_mov_b32 s15, 0x1003
s_cmp_lg_u32 s6, 0
s_cbranch_scc0 counts_ready
s_mov_b32 s15, 0x2004
counts_ready:
s_bfe_u32 s16, s15, 0xc0000
s_lshr_b32 s19, s15, 12
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
s_mov_b64 s[22:23], exec
v_cmpx_eq_u32 0, v0
s_cbranch_execz finish_alloc
s_mov_b32 m0, s15
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
finish_alloc:
s_mov_b64 exec, s[22:23]
s_barrier
v_cmpx_gt_u32 s16, v0
v_and_b32_e32 v10, 1, v0
v_and_b32_e32 v11, 2, v0
v_cmp_eq_u32_e64 s[20:21], 1, v10
v_cmp_eq_u32_e64 s[12:13], 2, v11
v_cndmask_b32_e64 v1, -1.0, 1.0, s[20:21]
v_cndmask_b32_e64 v2, -1.0, 1.0, s[12:13]
v_cndmask_b32_e64 v5, 0, 1.0, s[20:21]
v_cndmask_b32_e64 v6, 0, 1.0, s[12:13]
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v7, 0.5
exp pos0 v1, v2, v3, v4 done
exp param0 v5, v6, v7, v4
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32 s19, v0
v_cmp_eq_u32_e64 s[20:21], 1, v0
v_mov_b32_e32 v9, 0x200400
v_cndmask_b32_e64 v9, v9, s14, s[20:21]
exp prim v9, off, off, off done
s_endpgm
