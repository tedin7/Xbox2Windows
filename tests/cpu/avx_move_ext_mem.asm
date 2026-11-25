; coverage: source cpu_aot_avx_move_ext_mem
; Source-built MASM fixture exercising opt-in memory-operand forms (--mem-ops) of all 20
; register-domain AVX move, shuffle, permute, unpack, blend, and lane ops.
;   RCX = const void* a (96 bytes)
;   RDX = const void* b (96 bytes)
;   R8  = void*       c (96 bytes; [0..31] is output, [32..95] may hold inputs)
.code

op_vblendpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vblendpd xmm2, xmm1, XMMWORD PTR [rdx+16], 2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vblendpd_xmm_mem ENDP

op_vblendpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vblendpd ymm0, ymm1, YMMWORD PTR [rdx+32], 5
ELSE
    vblendpd ymm0, ymm1, YMMWORD PTR [rdx+32], 0Ah
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vblendpd_ymm_mem ENDP

op_vblendps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vblendps xmm2, xmm1, XMMWORD PTR [rcx+32], 0Ah
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vblendps_xmm_mem ENDP

op_vblendps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vblendps ymm0, ymm1, YMMWORD PTR [rdx+16], 5Ah
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vblendps_ymm_mem ENDP

op_vblendvpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu xmm3, XMMWORD PTR [rdx+32]
    vblendvpd xmm2, xmm1, XMMWORD PTR [rcx+48], xmm3
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vblendvpd_xmm_mem ENDP

op_vblendvpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm3, YMMWORD PTR [rdx+32]
    vblendvpd ymm0, ymm1, YMMWORD PTR [rdx], ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vblendvpd_ymm_mem ENDP

op_vblendvps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu xmm3, XMMWORD PTR [rcx+48]
    vblendvps xmm2, xmm1, XMMWORD PTR [rdx+16], xmm3
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vblendvps_xmm_mem ENDP

op_vblendvps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm3, YMMWORD PTR [rdx+32]
    vblendvps ymm0, ymm1, YMMWORD PTR [rcx+32], ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vblendvps_ymm_mem ENDP

op_vextractf128_mem_lo PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextractf128 XMMWORD PTR [r8], ymm0, 0
    vzeroupper
    ret
op_vextractf128_mem_lo ENDP

op_vextractf128_mem_hi PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextractf128 XMMWORD PTR [r8], ymm0, 1
    vzeroupper
    ret
op_vextractf128_mem_hi ENDP

op_vextractf128_mem_disp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vextractf128 XMMWORD PTR [r8+16], ymm0, 0
    vextractf128 XMMWORD PTR [r8], ymm0, 1
    vzeroupper
    ret
op_vextractf128_mem_disp ENDP

op_vinsertf128_mem_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vinsertf128 ymm0, ymm1, XMMWORD PTR [rdx+3], 0
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vinsertf128_mem_lo ENDP

op_vinsertf128_mem_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vinsertf128 ymm0, ymm1, XMMWORD PTR [rdx+19], 1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vinsertf128_mem_hi ENDP

op_vextractps_mem PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vextractps DWORD PTR [r8], xmm0, 2
    vextractps DWORD PTR [r8+4], xmm0, 3
    vzeroupper
    ret
op_vextractps_mem ENDP

op_vinsertps_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vinsertps xmm0, xmm1, DWORD PTR [rdx+7], 0D2h
ELSE
    ; imm8=0D4h: src_sel=3 (ignored for m32!), dst_sel=1, zmask=4.
    vinsertps xmm0, xmm1, DWORD PTR [rdx+7], 0D4h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vinsertps_mem ENDP

op_vperm2f128_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vperm2f128 ymm0, ymm1, YMMWORD PTR [rdx+16], 31h
ELSE
    vperm2f128 ymm0, ymm1, YMMWORD PTR [rdx+16], 20h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vperm2f128_mem ENDP

op_vpermilpd_imm_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpermilpd xmm0, XMMWORD PTR [rcx+3], 1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilpd_imm_xmm_mem ENDP

op_vpermilpd_imm_ymm_mem PROC FRAME
    .endprolog
    vpermilpd ymm0, YMMWORD PTR [rcx+3], 5
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilpd_imm_ymm_mem ENDP

op_vpermilpd_var_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilpd ymm0, ymm1, YMMWORD PTR [rdx+8]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilpd_var_ymm_mem ENDP

op_vpermilps_imm_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vpermilps xmm0, XMMWORD PTR [rcx+5], 1Bh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilps_imm_xmm_mem ENDP

op_vpermilps_imm_ymm_mem PROC FRAME
    .endprolog
    vpermilps ymm0, YMMWORD PTR [rcx+5], 93h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilps_imm_ymm_mem ENDP

op_vpermilps_var_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilps ymm0, ymm1, YMMWORD PTR [rdx+8]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpermilps_var_ymm_mem ENDP

op_vshufpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vshufpd xmm0, xmm1, XMMWORD PTR [rdx+16], 2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vshufpd_xmm_mem ENDP

op_vshufpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vshufpd ymm0, ymm1, YMMWORD PTR [rdx+16], 9
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vshufpd_ymm_mem ENDP

op_vshufps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vshufps xmm0, xmm1, XMMWORD PTR [rdx+16], 0E4h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vshufps_xmm_mem ENDP

op_vshufps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vshufps ymm0, ymm1, YMMWORD PTR [rdx+16], 4Eh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vshufps_ymm_mem ENDP

op_vunpcklpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vunpcklpd xmm0, xmm1, XMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpcklpd_xmm_mem ENDP

op_vunpcklpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vunpcklpd ymm0, ymm1, YMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpcklpd_ymm_mem ENDP

op_vunpckhpd_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vunpckhpd xmm0, xmm1, XMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpckhpd_xmm_mem ENDP

op_vunpckhpd_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vunpckhpd ymm0, ymm1, YMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpckhpd_ymm_mem ENDP

op_vunpcklps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vunpcklps xmm0, xmm1, XMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpcklps_xmm_mem ENDP

op_vunpcklps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vunpcklps ymm0, ymm1, YMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpcklps_ymm_mem ENDP

op_vunpckhps_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vunpckhps xmm0, xmm1, XMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpckhps_xmm_mem ENDP

op_vunpckhps_ymm_mem PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vunpckhps ymm0, ymm1, YMMWORD PTR [rdx+16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vunpckhps_ymm_mem ENDP

op_vmovddup_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovddup xmm0, QWORD PTR [rcx+3]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovddup_xmm_mem ENDP

op_vmovddup_ymm_mem PROC FRAME
    .endprolog
    vmovddup ymm0, YMMWORD PTR [rcx+3]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovddup_ymm_mem ENDP

op_vmovshdup_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovshdup xmm0, XMMWORD PTR [rcx+5]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovshdup_xmm_mem ENDP

op_vmovshdup_ymm_mem PROC FRAME
    .endprolog
    vmovshdup ymm0, YMMWORD PTR [rcx+5]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovshdup_ymm_mem ENDP

op_vmovsldup_xmm_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovsldup xmm0, XMMWORD PTR [rcx+7]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovsldup_xmm_mem ENDP

op_vmovsldup_ymm_mem PROC FRAME
    .endprolog
    vmovsldup ymm0, YMMWORD PTR [rcx+7]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovsldup_ymm_mem ENDP

; Exact width + discarded-read fault test: m64 vmovddup and m32 vinsertps must not overread,
; and a discarded read from [rdx] must still fault after storing the low 16 bytes to [r8].
op_discarded_mem_read PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu XMMWORD PTR [r8], xmm0
    vinsertps xmm1, xmm0, DWORD PTR [rdx], 0D4h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_discarded_mem_read ENDP

END
