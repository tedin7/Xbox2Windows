; coverage: source cpu_aot_sse_fp_logic
; Source-built MASM fixture exercising legacy SSE packed bitwise FP logic forms
; from docs/coverage/x64_zen2.tsv group SSE: andps, andnps, orps, xorps (xmm and
; aligned m128 sources, plus self-zeroing xorps xmmN, xmmN).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned; second vector at +16)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

; --- ANDPS ---
op_andps PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    orps xmm0, xmm1
ELSE
    andps xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andps ENDP

op_andps_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andps xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andps_mem ENDP

op_andps_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andps xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andps_mem_disp ENDP

op_andps_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andps xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andps_self ENDP

; --- ANDNPS ((~dst) & src) ---
op_andnps PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    andnps xmm1, xmm0
    movaps XMMWORD PTR [r8], xmm1
ELSE
    andnps xmm0, xmm1
    movaps XMMWORD PTR [r8], xmm0
ENDIF
    ret
op_andnps ENDP

op_andnps_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    andps xmm0, XMMWORD PTR [rdx]
ELSE
    andnps xmm0, XMMWORD PTR [rdx]
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnps_mem ENDP

op_andnps_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andnps xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnps_mem_disp ENDP

op_andnps_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andnps xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_andnps_self ENDP

; --- ORPS ---
op_orps PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    andps xmm0, xmm1
ELSE
    orps xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orps ENDP

op_orps_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orps xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orps_mem ENDP

op_orps_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orps xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orps_mem_disp ENDP

op_orps_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    orps xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_orps_self ENDP

; --- XORPS (binary & self-zeroing) ---
op_xorps PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    orps xmm0, xmm1
ELSE
    xorps xmm0, xmm1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorps ENDP

op_xorps_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorps xmm0, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorps_mem ENDP

op_xorps_mem_disp PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorps xmm0, XMMWORD PTR [rdx+16]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorps_mem_disp ENDP

op_xorps_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    xorps xmm0, xmm0
    movaps XMMWORD PTR [r8], xmm0
    ret
op_xorps_self ENDP

; --- Self-zeroing on undefined XMM registers feeding bitwise + FP arithmetic ---
op_xorps_zero_logic PROC FRAME
    .endprolog
    xorps xmm3, xmm3
    movaps xmm1, XMMWORD PTR [rcx]
    andnps xmm3, xmm1
    movaps xmm2, XMMWORD PTR [rdx]
    orps xmm3, xmm2
    movaps XMMWORD PTR [r8], xmm3
    ret
op_xorps_zero_logic ENDP

op_xorps_zero_arith PROC FRAME
    .endprolog
    xorps xmm4, xmm4
    addps xmm4, XMMWORD PTR [rcx]
    mulps xmm4, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm4
    ret
op_xorps_zero_arith ENDP

; --- Chained logic + FP pipeline with [rcx+16] aligned load ---
op_logic_pipeline PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
    movaps xmm2, XMMWORD PTR [rcx+16]
    andps xmm0, xmm1
IFDEF ALTER_IMM
    andps xmm2, XMMWORD PTR [rdx+16]
ELSE
    andnps xmm2, XMMWORD PTR [rdx+16]
ENDIF
    xorps xmm5, xmm5
    orps xmm5, xmm0
    xorps xmm5, xmm2
    addps xmm5, xmm1
    movaps XMMWORD PTR [r8], xmm5
    subps xmm0, xmm2
    movaps XMMWORD PTR [r8+16], xmm0
    ret
op_logic_pipeline ENDP

; --- Discarded aligned m128 read must still fault on NULL rdx even under /O2 ---
op_logic_discarded_read PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    andps xmm0, XMMWORD PTR [rdx]
    movaps xmm1, XMMWORD PTR [rcx]
    movaps XMMWORD PTR [r8], xmm1
    ret
op_logic_discarded_read ENDP

END
