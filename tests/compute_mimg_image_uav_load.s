s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
v_lshl_add_u32 v8, s10, 2, v0
v_lshl_add_u32 v9, s11, 1, v1
s_waitcnt lgkmcnt(0)
image_load v[14:17], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
v_mov_b32_e32 v18, v15
v_mov_b32_e32 v19, v16
v_mov_b32_e32 v20, v17
v_mov_b32_e32 v21, v14
image_store v[18:21], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D
s_endpgm
