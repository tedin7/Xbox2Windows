; coverage: record cpu_aot_sse_fp_pd_moves_cmp
; Source-built MASM fixture exercising legacy SSE2 scalar double compare (cmpsd),
; half-qword double moves (movlpd, movhpd) and aligned non-temporal double store
; (movntpd) from docs/coverage/x64_zen2.tsv group SSE2.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned; second vector at +16)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

; --- cmpsd: all eight predicates, register/m64/unaligned +1 sources, bits 127:64 preserved ---

op_cmpsd_0 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 4
ELSE
    cmpsd xmm0, xmm1, 0
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_0 ENDP

op_cmpsd_1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 5
ELSE
    cmpsd xmm0, xmm1, 1
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_1 ENDP

op_cmpsd_2 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 6
ELSE
    cmpsd xmm0, xmm1, 2
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_2 ENDP

op_cmpsd_3 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 7
ELSE
    cmpsd xmm0, xmm1, 3
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_3 ENDP

op_cmpsd_4 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 0
ELSE
    cmpsd xmm0, xmm1, 4
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_4 ENDP

op_cmpsd_5 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 1
ELSE
    cmpsd xmm0, xmm1, 5
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_5 ENDP

op_cmpsd_6 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 2
ELSE
    cmpsd xmm0, xmm1, 6
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_6 ENDP

op_cmpsd_7 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 3
ELSE
    cmpsd xmm0, xmm1, 7
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_7 ENDP

op_cmpsd_mem PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    cmpsd xmm0, QWORD PTR [rdx+16], 5
ELSE
    cmpsd xmm0, QWORD PTR [rdx], 5
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_mem ENDP

op_cmpsd_mem_plus1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    cmpsd xmm0, QWORD PTR [rdx+1], 6
ELSE
    cmpsd xmm0, QWORD PTR [rdx+1], 2
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_mem_plus1 ENDP

op_cmpsd_self PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 0
ELSE
    cmpsd xmm0, xmm0, 0
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_cmpsd_self ENDP

; --- movlpd / movhpd: m64 load-merge (preserving other 64 bits) and m64 store ---

op_movlpd_load PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    movhpd xmm0, QWORD PTR [rdx]
ELSE
    movlpd xmm0, QWORD PTR [rdx]
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_movlpd_load ENDP

op_movlpd_store PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    movhpd QWORD PTR [r8+8], xmm0
ELSE
    movlpd QWORD PTR [r8+8], xmm0
ENDIF
    ret
op_movlpd_store ENDP

op_movhpd_load PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    movlpd xmm0, QWORD PTR [rdx]
ELSE
    movhpd xmm0, QWORD PTR [rdx]
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_movhpd_load ENDP

op_movhpd_store PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    movlpd QWORD PTR [r8+8], xmm0
ELSE
    movhpd QWORD PTR [r8+8], xmm0
ENDIF
    ret
op_movhpd_store ENDP

op_movlpd_load_plus1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movlpd xmm0, QWORD PTR [rdx+1]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_movlpd_load_plus1 ENDP

op_movhpd_load_plus1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movhpd xmm0, QWORD PTR [rdx+1]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_movhpd_load_plus1 ENDP

op_movlpd_store_plus1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movlpd QWORD PTR [r8+9], xmm0
    ret
op_movlpd_store_plus1 ENDP

op_movhpd_store_plus1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movhpd QWORD PTR [r8+9], xmm0
    ret
op_movhpd_store_plus1 ENDP

; --- movntpd: 16-byte aligned non-temporal store ---

op_movntpd PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    movntpd XMMWORD PTR [r8+16], xmm0
ELSE
    movntpd XMMWORD PTR [r8], xmm0
ENDIF
    ret
op_movntpd ENDP

op_movntpd_disp16 PROC FRAME
    .endprolog
    movdqu xmm0, XMMWORD PTR [rcx+1]
    movntpd XMMWORD PTR [r8+16], xmm0
    ret
op_movntpd_disp16 ENDP

; --- Chained pipeline and discarded faulting m64 read ---

op_pd_moves_cmp_pipeline PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movdqu xmm1, XMMWORD PTR [rdx]
    movdqa xmm2, XMMWORD PTR [rdx+16]
    movlpd xmm0, QWORD PTR [rdx+8]
    movhpd xmm2, QWORD PTR [rcx+24]
IFDEF ALTER_IMM
    cmpsd xmm0, xmm1, 0
ELSE
    cmpsd xmm0, xmm1, 4
ENDIF
    cmpsd xmm2, QWORD PTR [rdx+4], 2
    movntpd XMMWORD PTR [r8], xmm0
    movhpd QWORD PTR [r8+16], xmm2
    movlpd QWORD PTR [r8+24], xmm2
    ret
op_pd_moves_cmp_pipeline ENDP

op_pd_moves_cmp_discarded_read PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm5, XMMWORD PTR [rcx+16]
    movlpd xmm5, QWORD PTR [rdx]
    movaps XMMWORD PTR [r8], xmm0
    ret
op_pd_moves_cmp_discarded_read ENDP

END
