; coverage: record cpu_aot_avx_test
; Source-owned differential fixture for AVX test-flag instructions:
; VPTEST, VTESTPS, and VTESTPD across 128-bit (XMM) and 256-bit (YMM) register and memory forms.
; Each export returns (CF << 0) | (ZF << 1) in RAX.
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       scratch (32 bytes)
.code

PACK_FLAGS MACRO
    setb al
    sete r9b
    movzx eax, al
    movzx r9d, r9b
    shl r9d, 1
    or eax, r9d
ENDM

op_vptest_xmm_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vptest xmm1, xmm0
ELSE
    vptest xmm0, xmm1
ENDIF
    PACK_FLAGS
    ret
op_vptest_xmm_reg ENDP

op_vptest_xmm_mem PROC FRAME
    .endprolog
    vmovdqu xmm2, XMMWORD PTR [rcx]
    vptest xmm2, XMMWORD PTR [rdx]
    PACK_FLAGS
    ret
op_vptest_xmm_mem ENDP

op_vptest_ymm_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vptest ymm1, ymm0
ELSE
    vptest ymm0, ymm1
ENDIF
    vzeroupper
    PACK_FLAGS
    ret
op_vptest_ymm_reg ENDP

op_vptest_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vptest ymm2, YMMWORD PTR [rdx]
    vzeroupper
    PACK_FLAGS
    ret
op_vptest_ymm_mem ENDP

op_vtestps_xmm_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vptest xmm0, xmm1
ELSE
    vtestps xmm0, xmm1
ENDIF
    PACK_FLAGS
    ret
op_vtestps_xmm_reg ENDP

op_vtestps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, XMMWORD PTR [rcx]
    vtestps xmm3, XMMWORD PTR [rdx]
    PACK_FLAGS
    ret
op_vtestps_xmm_mem ENDP

op_vtestps_ymm_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vptest ymm0, ymm1
ELSE
    vtestps ymm0, ymm1
ENDIF
    vzeroupper
    PACK_FLAGS
    ret
op_vtestps_ymm_reg ENDP

op_vtestps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm3, YMMWORD PTR [rcx]
    vtestps ymm3, YMMWORD PTR [rdx]
    vzeroupper
    PACK_FLAGS
    ret
op_vtestps_ymm_mem ENDP

op_vtestpd_xmm_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vtestps xmm0, xmm1
ELSE
    vtestpd xmm0, xmm1
ENDIF
    PACK_FLAGS
    ret
op_vtestpd_xmm_reg ENDP

op_vtestpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu xmm4, XMMWORD PTR [rcx]
    vtestpd xmm4, XMMWORD PTR [rdx]
    PACK_FLAGS
    ret
op_vtestpd_xmm_mem ENDP

op_vtestpd_ymm_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
IFDEF ALTER_TEST
    vtestps ymm0, ymm1
ELSE
    vtestpd ymm0, ymm1
ENDIF
    vzeroupper
    PACK_FLAGS
    ret
op_vtestpd_ymm_reg ENDP

op_vtestpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vtestpd ymm4, YMMWORD PTR [rdx]
    vzeroupper
    PACK_FLAGS
    ret
op_vtestpd_ymm_mem ENDP

; VEX.128 load into XMM0 must zero YMM0[255:128] before a 256-bit VPTEST.
op_vptest_ymm_after_xmm_zero PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vptest ymm0, ymm1
    vzeroupper
    PACK_FLAGS
    ret
op_vptest_ymm_after_xmm_zero ENDP

; VEX.128 load into XMM0 must zero YMM0[255:128] before a 256-bit VTESTPS.
op_vtestps_ymm_after_xmm_zero PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vtestps ymm0, ymm1
    vzeroupper
    PACK_FLAGS
    ret
op_vtestps_ymm_after_xmm_zero ENDP

; Self-test: CF is always 1 ((~a & a) == 0); ZF is 1 iff a == 0.
op_vtest_self PROC FRAME
    .endprolog
    vmovdqu ymm5, YMMWORD PTR [rcx]
    vptest ymm5, ymm5
    vzeroupper
    PACK_FLAGS
    ret
op_vtest_self ENDP

; Store-reload via [r8] and scratch GPRs (xor/mov/setb/sete/shl/or).
op_vtest_r8_store_reload PROC FRAME
    .endprolog
    xor r10d, r10d
    xor r11d, r11d
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vtestpd ymm1, YMMWORD PTR [r8]
    vzeroupper
    setb r10b
    sete r11b
    shl r11d, 1
    or r10d, r11d
    mov eax, r10d
    ret
op_vtest_r8_store_reload ENDP

END
