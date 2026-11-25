s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
v_lshl_add_u32 v8, s10, 2, v0
v_lshl_add_u32 v9, s11, 1, v1
v_mad_u32_u24 v10, 8, v9, v8
v_lshlrev_b32_e32 v11, 16, v10
v_or_b32_e32 v12, v11, v10
v_or_b32_e32 v14, 0xc0003c00, v12
v_or_b32_e32 v15, 0x80000000, v12
s_waitcnt lgkmcnt(0)
image_store v[14:15], v[8:9], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D d16
s_endpgm
