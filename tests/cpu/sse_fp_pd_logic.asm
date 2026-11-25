; coverage: source cpu_aot_sse_fp_pd_logic
; Source-built MASM fixture exercising legacy SSE2 packed double bitwise logic forms
; from docs/coverage/x64_zen2.tsv group SSE2: andpd, andnpd, orpd, xorpd (xmm and
; aligned m128 sources, plus self-zeroing xorpd xmmN, xmmN).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned; second vector at +16)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

; --- ANDPD ---
op_andpd PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    orpd xmm0, xmm1
ELSE
    andpd xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andpd ENDP

op_andpd_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andpd xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andpd_mem ENDP

op_andpd_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andpd xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andpd_mem_disp ENDP

op_andpd_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andpd xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andpd_self ENDP

; --- ANDNPD ((~dst) & src) ---
op_andnpd PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    andnpd xmm1, xmm0
    movaps XMMWORD PTR [r8], xmm1
ELSE
    andnpd xmm0, xmm1
    movaps XMMWORD PTR [r8], xmm0
ENDIF
    ret
op_andnpd ENDP

op_andnpd_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    andpd xmm0, XMMWORD PTR [rdx]
ELSE
    andnpd xmm0, XMMWORD PTR [rdx]
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnpd_mem ENDP

op_andnpd_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andnpd xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnpd_mem_disp ENDP

op_andnpd_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andnpd xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnpd_self ENDP

; --- ORPD ---
op_orpd PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    andpd xmm0, xmm1
ELSE
    orpd xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orpd ENDP

op_orpd_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orpd xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orpd_mem ENDP

op_orpd_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orpd xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orpd_mem_disp ENDP

op_orpd_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orpd xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orpd_self ENDP

; --- XORPD (binary & self-zeroing) ---
op_xorpd PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    orpd xmm0, xmm1
ELSE
    xorpd xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorpd ENDP

op_xorpd_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorpd xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorpd_mem ENDP

op_xorpd_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorpd xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorpd_mem_disp ENDP

op_xorpd_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorpd xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorpd_self ENDP

; --- Self-zeroing on undefined XMM registers feeding bitwise + double arithmetic ---
op_xorpd_zero_logic PROC FRAME
    .endprolog
    xorpd xmm3, xmm3
    movaps xmm1, XMMWORD PTR [rcx]
    andnpd xmm3, xmm1
    movaps xmm2, XMMWORD PTR [rdx]
    orpd xmm3, xmm2
    movaps XMMWORD PTR [r8], xmm3
    ret
op_xorpd_zero_logic ENDP

op_xorpd_zero_arith PROC FRAME
    .endprolog
    xorpd xmm4, xmm4
    addpd xmm4, XMMWORD PTR [rcx]
    mulpd xmm4, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm4
    ret
op_xorpd_zero_arith ENDP

; --- Chained logic + double FP pipeline with [rcx+16] aligned load ---
op_pd_logic_pipeline PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
    movaps xmm2, XMMWORD PTR [rcx+16]
    andpd xmm0, xmm1
IFDEF ALTER_IMM
    andpd xmm2, XMMWORD PTR [rdx+16]
ELSE
    andnpd xmm2, XMMWORD PTR [rdx+16]
ENDIF
    xorpd xmm5, xmm5
    orpd xmm5, xmm0
    xorpd xmm5, xmm2
    addpd xmm5, xmm1
    movaps XMMWORD PTR [r8], xmm5
    subpd xmm0, xmm2
    movaps XMMWORD PTR [r8+16], xmm0
    ret
op_pd_logic_pipeline ENDP

; --- Discarded aligned m128 read must still fault on NULL rdx even under /O2 ---
op_pd_logic_discarded_read PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andpd xmm0, XMMWORD PTR [rdx]
    movaps xmm1, XMMWORD PTR [rcx]
    movaps XMMWORD PTR [r8], xmm1
    ret
op_pd_logic_discarded_read ENDP

END
