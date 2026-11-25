; Source-built mesh for two wave64s that exports two clip distances and one cull distance.
; PA_CL_VS_OUT_CNTL 0x00c01003 enables CLIP_DIST_ENA_0, CLIP_DIST_ENA_1, CULL_DIST_ENA_4 and the
; CCDIST0 and CCDIST1 vectors: pos1 carries components 0-3 and pos2 components 4-7.
; Geometry dword 0 is the allocation word: 0x2006 (6 vertices, 2 triangles) draws, any other
; value allocates nothing. Geometry dword 1 selects the pair of triangles: 6 draws the culling
; pair C and D, any other value the clipping pair A and B. Thread i exports vertex i of the pair
; and triangle (3i, 3i + 1, 3i + 2).
;   A (0-2):  clip 0 crosses zero at screen x = 9: the triangle is partly visible.
;   B (3-5):  clip 1 crosses zero at screen y = 7: the triangle is partly visible.
;   C (6-8):  cull 4 is -1 on every vertex, so the whole triangle is culled.
;   D (9-11): cull 4 is +1 on every vertex, so the neighbouring triangle stays.
; Components the word does not enable hold -1.0, except component 5 (pos2.y), which holds 2.0.
.text
s_mov_b32 s10, s17
s_mov_b32 s11, 0
s_load_dwordx4 s[24:27], s[10:11], 0x20
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s15, s[24:27], null
s_buffer_load_dword s14, s[24:27], 4
s_waitcnt lgkmcnt(0)
s_cmp_eq_u32 s15, 0x2006
s_cbranch_scc0 suppress
s_cmp_eq_u32 s14, 6
s_cbranch_scc0 clip_pair
s_mov_b32 s9, 6
s_branch counts_ready
clip_pair:
s_mov_b32 s9, 0
s_branch counts_ready
suppress:
s_mov_b32 s15, 0
s_mov_b32 s9, 0
counts_ready:
s_bfe_u32 s16, s15, 0xc0000
s_lshr_b32 s19, s15, 12
s_mov_b32 s40, 0x300c03
s_mov_b32 s41, 0x200400
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
v_lshl_add_u32 v11, s9, 0, v0
s_mov_b64 s[22:23], exec
v_cmpx_eq_u32_e32 0, v0
s_cbranch_execz finish_alloc
s_mov_b32 m0, s15
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
finish_alloc:
s_mov_b64 exec, s[22:23]
s_barrier
v_mov_b32_e32 v3, 0.5
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 1.0
v_mov_b32_e32 v6, 1.0
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v8, -1.0
v_mov_b32_e32 v10, 2.0
v_cmpx_eq_u32_e32 0, v11
v_mov_b32_e32 v1, 0xbf600000
v_mov_b32_e32 v2, 0x3f600000
v_mov_b32_e32 v5, 0x40e00000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 1, v11
v_mov_b32_e32 v1, 0xbde00000
v_mov_b32_e32 v2, 0x3f600000
v_mov_b32_e32 v5, 0xc0a80000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 2, v11
v_mov_b32_e32 v1, 0xbf600000
v_mov_b32_e32 v2, 0x3de00000
v_mov_b32_e32 v5, 0x40e00000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 3, v11
v_mov_b32_e32 v1, 0x3e000000
v_mov_b32_e32 v2, 0x3f600000
v_mov_b32_e32 v6, 0xc0a00000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 4, v11
v_mov_b32_e32 v1, 0x3f640000
v_mov_b32_e32 v2, 0x3f600000
v_mov_b32_e32 v6, 0xc0a00000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 5, v11
v_mov_b32_e32 v1, 0x3e000000
v_mov_b32_e32 v2, 0x3de00000
v_mov_b32_e32 v6, 0x40e80000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 6, v11
v_mov_b32_e32 v1, 0xbf600000
v_mov_b32_e32 v2, 0xbe000000
v_mov_b32_e32 v7, -1.0
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 7, v11
v_mov_b32_e32 v1, 0xbde00000
v_mov_b32_e32 v2, 0xbe000000
v_mov_b32_e32 v7, -1.0
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 8, v11
v_mov_b32_e32 v1, 0xbf600000
v_mov_b32_e32 v2, 0xbf640000
v_mov_b32_e32 v7, -1.0
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 9, v11
v_mov_b32_e32 v1, 0x3e000000
v_mov_b32_e32 v2, 0xbe000000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 10, v11
v_mov_b32_e32 v1, 0x3f640000
v_mov_b32_e32 v2, 0xbe000000
s_mov_b64 exec, s[22:23]
v_cmpx_eq_u32_e32 11, v11
v_mov_b32_e32 v1, 0x3e000000
v_mov_b32_e32 v2, 0xbf640000
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32_e32 s16, v0
exp pos0 v1, v2, v3, v4
exp pos1 v5, v6, v8, v8
exp pos2 v7, v10, v8, v8 done
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32_e32 s19, v0
v_mad_u32_u24 v9, v0, s40, s41
exp prim v9, off, off, off done
s_endpgm
