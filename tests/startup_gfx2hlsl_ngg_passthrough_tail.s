s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_mbcnt_lo_u32_b32 v23, -1, 0
v_mbcnt_hi_u32_b32 v23, -1, v23
v_mul_u32_u24 v24, 48, v23
v_add_nc_u32 v25, 64, v24
ds_write_b32 v25, v11
ds_write_b32 v25, v12 offset:4
ds_write_b32 v25, v13 offset:8
ds_write_b32 v25, v14 offset:12
ds_write_b32 v25, v15 offset:16
ds_write_b32 v25, v16 offset:20
ds_write_b32 v25, v17 offset:24
ds_write_b32 v25, v18 offset:28
ds_write_b32 v25, v19 offset:32
ds_write_b32 v25, v20 offset:36
ds_write_b32 v25, v21 offset:40
ds_write_b32 v25, v22 offset:44
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v30, v24 offset:64
ds_read_b32 v31, v24 offset:68
ds_read_b32 v32, v24 offset:72
ds_read_b32 v33, v24 offset:76
ds_read_b32 v34, v24 offset:80
ds_read_b32 v35, v24 offset:84
ds_read_b32 v36, v24 offset:88
ds_read_b32 v37, v24 offset:92
ds_read_b32 v38, v24 offset:96
ds_read_b32 v39, v24 offset:100
ds_read_b32 v40, v24 offset:104
ds_read_b32 v41, v24 offset:108
s_waitcnt lgkmcnt(0)
exp pos0 v1, v2, v5, v4 done
exp param0 v30, v31, v32, v33
exp param1 v34, v35, v36, v37
exp param2 v38, v39, v40, v41
exp prim v0, off, off, off done
s_endpgm
