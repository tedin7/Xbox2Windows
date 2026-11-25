; coverage: source cpu_aot_avx2_mem
; Source-built MASM fixture exercising opt-in memory-operand forms (--mem-ops) of all 17
; register-domain AVX2 instructions across XMM and YMM widths, unaligned displacements,
; YMM->XMM zero-extension, and discarded-read ordering.
;   RCX = const void* input_a
;   RDX = const void* input_b
;   R8  = void*       output (32 bytes)
.code

op_extracti128_mem_lo PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextracti128 XMMWORD PTR [r8], ymm0, 0FEh
    vzeroupper
    ret
op_extracti128_mem_lo ENDP

op_extracti128_mem_hi PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextracti128 XMMWORD PTR [r8], ymm0, 0FFh
    vzeroupper
    ret
op_extracti128_mem_hi ENDP

op_extracti128_mem_disp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextracti128 XMMWORD PTR [r8 + 16], ymm0, 0
    vextracti128 XMMWORD PTR [r8], ymm0, 1
    vzeroupper
    ret
op_extracti128_mem_disp ENDP

op_inserti128_mem_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vinserti128 ymm0, ymm1, XMMWORD PTR [rdx], 0FEh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_inserti128_mem_lo ENDP

op_inserti128_mem_hi_disp PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vinserti128 ymm0, ymm1, XMMWORD PTR [rdx + 3], 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_inserti128_mem_hi_disp ENDP

op_perm2i128_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vperm2i128 ymm0, ymm1, YMMWORD PTR [rdx], 31h
ELSE
    vperm2i128 ymm0, ymm1, YMMWORD PTR [rdx], 20h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2i128_mem ENDP

op_pbroadcastb_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpbroadcastb xmm0, BYTE PTR [rcx + 3]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastb_xmm_mem ENDP

op_pbroadcastb_ymm_mem PROC FRAME
    .endprolog
    vpbroadcastb ymm0, BYTE PTR [rcx + 3]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastb_ymm_mem ENDP

op_pbroadcastw_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpbroadcastw xmm0, WORD PTR [rcx + 5]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastw_xmm_mem ENDP

op_pbroadcastw_ymm_mem PROC FRAME
    .endprolog
    vpbroadcastw ymm0, WORD PTR [rcx + 5]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastw_ymm_mem ENDP

op_pbroadcastd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpbroadcastd xmm0, DWORD PTR [rcx + 7]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastd_xmm_mem ENDP

op_pbroadcastd_ymm_mem PROC FRAME
    .endprolog
    vpbroadcastd ymm0, DWORD PTR [rcx + 7]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastd_ymm_mem ENDP

op_pbroadcastq_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpbroadcastq xmm0, QWORD PTR [rcx + 9]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastq_xmm_mem ENDP

op_pbroadcastq_ymm_mem PROC FRAME
    .endprolog
    vpbroadcastq ymm0, QWORD PTR [rcx + 9]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pbroadcastq_ymm_mem ENDP

op_permd_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permd_mem ENDP

op_permps_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermps ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permps_mem ENDP

op_permq_mem PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    vpermq ymm0, YMMWORD PTR [rcx], 4Eh
ELSE
    vpermq ymm0, YMMWORD PTR [rcx], 1Bh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permq_mem ENDP

op_permpd_mem PROC FRAME
    .endprolog
    vpermpd ymm0, YMMWORD PTR [rcx], 93h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permpd_mem ENDP

op_pblendd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpblendd xmm0, xmm1, XMMWORD PTR [rdx], 0FAh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendd_xmm_mem ENDP

op_pblendd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vpblendd ymm0, ymm1, YMMWORD PTR [rdx], 55h
ELSE
    vpblendd ymm0, ymm1, YMMWORD PTR [rdx], 0AAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendd_ymm_mem ENDP

op_psllvd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsllvd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllvd_xmm_mem ENDP

op_psllvd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpsllvd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllvd_ymm_mem ENDP

op_psllvq_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsllvq xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllvq_xmm_mem ENDP

op_psllvq_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpsllvq ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllvq_ymm_mem ENDP

op_psravd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsravd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psravd_xmm_mem ENDP

op_psravd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpsravd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psravd_ymm_mem ENDP

op_psrlvd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrlvd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlvd_xmm_mem ENDP

op_psrlvd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpsrlvd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlvd_ymm_mem ENDP

op_psrlvq_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrlvq xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlvq_xmm_mem ENDP

op_psrlvq_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpsrlvq ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlvq_ymm_mem ENDP

; Discarded memory read must remain ordered after first store and before second store.
op_discarded_mem_read PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu XMMWORD PTR [r8], xmm0
    vpbroadcastd ymm1, DWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_discarded_mem_read ENDP

; Multi-register pipeline exercising memory sources and extracti128 store.
op_mem_pipeline PROC FRAME
    .endprolog
    vpbroadcastd ymm2, DWORD PTR [rdx + 7]
    vpermq ymm3, YMMWORD PTR [rcx], 0D8h
    vpblendd ymm4, ymm3, YMMWORD PTR [rdx], 5Ah
    vpsravd ymm5, ymm4, YMMWORD PTR [rdx]
    vinserti128 ymm5, ymm5, XMMWORD PTR [rcx + 3], 1
    vpermd ymm0, ymm2, ymm5
    vextracti128 XMMWORD PTR [r8 + 16], ymm0, 0
    vextracti128 XMMWORD PTR [r8], ymm0, 1
    vzeroupper
    ret
op_mem_pipeline ENDP

END
