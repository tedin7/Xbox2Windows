; coverage: source cpu_aot_sse_mmx_int
; Source-built MASM fixture exercising legacy MMX and SSE2 packed unsigned
; integer operations (docs/coverage/x64_zen2.tsv group SSE):
; pavgb, pavgw, pmulhuw, and psadbw.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

; --- MMX PAVGB (mm/m64; unaligned m64 allowed) ---
op_pavgb_mm_reg PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    movq mm1, QWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgw mm0, mm1
ELSE
    pavgb mm0, mm1
ENDIF
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pavgb_mm_reg ENDP

op_pavgb_mm_mem PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    pavgb mm0, QWORD PTR [rdx]
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pavgb_mm_mem ENDP

op_pavgb_mm_mem_plus1 PROC FRAME
    .endprolog
    movq mm2, QWORD PTR [rcx]
    pavgb mm2, QWORD PTR [rdx+1]
    movq QWORD PTR [r8], mm2
    emms
    ret
op_pavgb_mm_mem_plus1 ENDP

op_pavgb_mm_self PROC FRAME
    .endprolog
    movq mm3, QWORD PTR [rcx]
    pavgb mm3, mm3
    movq QWORD PTR [r8], mm3
    emms
    ret
op_pavgb_mm_self ENDP

; --- MMX PAVGW ---
op_pavgw_mm_reg PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    movq mm1, QWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgb mm0, mm1
ELSE
    pavgw mm0, mm1
ENDIF
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pavgw_mm_reg ENDP

op_pavgw_mm_mem PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    pavgw mm0, QWORD PTR [rdx]
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pavgw_mm_mem ENDP

op_pavgw_mm_mem_plus1 PROC FRAME
    .endprolog
    movq mm2, QWORD PTR [rcx]
    pavgw mm2, QWORD PTR [rdx+1]
    movq QWORD PTR [r8], mm2
    emms
    ret
op_pavgw_mm_mem_plus1 ENDP

op_pavgw_mm_self PROC FRAME
    .endprolog
    movq mm3, QWORD PTR [rcx]
    pavgw mm3, mm3
    movq QWORD PTR [r8], mm3
    emms
    ret
op_pavgw_mm_self ENDP

; --- MMX PMULHUW ---
op_pmulhuw_mm_reg PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    movq mm1, QWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgw mm0, mm1
ELSE
    pmulhuw mm0, mm1
ENDIF
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pmulhuw_mm_reg ENDP

op_pmulhuw_mm_mem PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    pmulhuw mm0, QWORD PTR [rdx]
    movq QWORD PTR [r8], mm0
    emms
    ret
op_pmulhuw_mm_mem ENDP

op_pmulhuw_mm_mem_plus1 PROC FRAME
    .endprolog
    movq mm2, QWORD PTR [rcx]
    pmulhuw mm2, QWORD PTR [rdx+1]
    movq QWORD PTR [r8], mm2
    emms
    ret
op_pmulhuw_mm_mem_plus1 ENDP

op_pmulhuw_mm_self PROC FRAME
    .endprolog
    movq mm3, QWORD PTR [rcx]
    pmulhuw mm3, mm3
    movq QWORD PTR [r8], mm3
    emms
    ret
op_pmulhuw_mm_self ENDP

; --- MMX PSADBW ---
op_psadbw_mm_reg PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    movq mm1, QWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgb mm0, mm1
ELSE
    psadbw mm0, mm1
ENDIF
    movq QWORD PTR [r8], mm0
    emms
    ret
op_psadbw_mm_reg ENDP

op_psadbw_mm_mem PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    psadbw mm0, QWORD PTR [rdx]
    movq QWORD PTR [r8], mm0
    emms
    ret
op_psadbw_mm_mem ENDP

op_psadbw_mm_mem_plus1 PROC FRAME
    .endprolog
    movq mm2, QWORD PTR [rcx]
    psadbw mm2, QWORD PTR [rdx+1]
    movq QWORD PTR [r8], mm2
    emms
    ret
op_psadbw_mm_mem_plus1 ENDP

op_psadbw_mm_self PROC FRAME
    .endprolog
    movq mm3, QWORD PTR [rcx]
    psadbw mm3, mm3
    movq QWORD PTR [r8], mm3
    emms
    ret
op_psadbw_mm_self ENDP

; --- SSE2 XMM PAVGB (16-byte aligned m128) ---
op_pavgb_xmm_reg PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgw xmm0, xmm1
ELSE
    pavgb xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgb_xmm_reg ENDP

op_pavgb_xmm_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgb xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgb_xmm_mem ENDP

op_pavgb_xmm_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgb xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgb_xmm_mem_disp ENDP

op_pavgb_xmm_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgb xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgb_xmm_self ENDP

; --- SSE2 XMM PAVGW ---
op_pavgw_xmm_reg PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgb xmm0, xmm1
ELSE
    pavgw xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgw_xmm_reg ENDP

op_pavgw_xmm_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgw xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgw_xmm_mem ENDP

op_pavgw_xmm_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgw xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgw_xmm_mem_disp ENDP

op_pavgw_xmm_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgw xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pavgw_xmm_self ENDP

; --- SSE2 XMM PMULHUW ---
op_pmulhuw_xmm_reg PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgw xmm0, xmm1
ELSE
    pmulhuw xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pmulhuw_xmm_reg ENDP

op_pmulhuw_xmm_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pmulhuw xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pmulhuw_xmm_mem ENDP

op_pmulhuw_xmm_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pmulhuw xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pmulhuw_xmm_mem_disp ENDP

op_pmulhuw_xmm_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pmulhuw xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pmulhuw_xmm_self ENDP

; --- SSE2 XMM PSADBW ---
op_psadbw_xmm_reg PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    pavgb xmm0, xmm1
ELSE
    psadbw xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_psadbw_xmm_reg ENDP

op_psadbw_xmm_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    psadbw xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_psadbw_xmm_mem ENDP

op_psadbw_xmm_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    psadbw xmm0, XMMWORD PTR [rdx]
ELSE
    psadbw xmm0, XMMWORD PTR [rdx+16]
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_psadbw_xmm_mem_disp ENDP

op_psadbw_xmm_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    psadbw xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_psadbw_xmm_self ENDP

; --- Combined MMX + XMM pipeline & discarded fault read ---
op_mmx_int_pipeline PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    pavgb mm0, QWORD PTR [rdx+1]
    pmulhuw mm0, QWORD PTR [rcx+8]
    movq QWORD PTR [r8], mm0
    movq mm1, QWORD PTR [rcx+8]
    psadbw mm1, QWORD PTR [rdx+1]
    movq QWORD PTR [r8+8], mm1
    emms

    movdqa xmm0, XMMWORD PTR [rcx]
    pavgw xmm0, XMMWORD PTR [rdx+16]
IFDEF ALTER_IMM
    pavgb xmm0, XMMWORD PTR [rdx]
ELSE
    psadbw xmm0, XMMWORD PTR [rdx]
ENDIF
    movaps XMMWORD PTR [r8+16], xmm0
    ret
op_mmx_int_pipeline ENDP

op_mmx_int_discarded_read PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    pavgb xmm0, XMMWORD PTR [rdx]
    movaps xmm1, XMMWORD PTR [rcx]
    movaps XMMWORD PTR [r8], xmm1
    ret
op_mmx_int_discarded_read ENDP

END
