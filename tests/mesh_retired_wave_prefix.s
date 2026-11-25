; Prepend to mesh_dynamic_geometry.s. Both waves reach the first barrier.
; Wave 1 then exits and wave 0 alone executes the mesh body.
.text
s_barrier
s_bfe_u32 s8, s3, 0x40018
s_cmp_eq_u32 s8, 0
s_cbranch_scc0 retiring
s_barrier
s_branch mesh_body
retiring:
s_branch mesh_exit
mesh_body:
