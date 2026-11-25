; coverage: source cpu_aot_mmx_sse2
; Source-built MASM fixture exercising the MMX/SSE2 conversion and bridge
; instructions from docs/coverage/x64_zen2.tsv: cvtpd2pi, cvttpd2pi, cvtpi2pd,
; movdq2q, movq2dq (SSE2) and emms (MMX), plus movdqa, movdqu and movq transport glue.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

op_cvtpd2pi_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    cvttpd2pi mm0, xmm0
ELSE
    cvtpd2pi mm0, xmm0
ENDIF
    movq QWORD PTR [r8], mm0
    emms
    ret
op_cvtpd2pi_reg ENDP

op_cvtpd2pi_mem PROC FRAME
    .endprolog
    cvtpd2pi mm1, XMMWORD PTR [rcx]
    movq QWORD PTR [r8], mm1
    emms
    ret
op_cvtpd2pi_mem ENDP

op_cvtpd2pi_mem_disp PROC FRAME
    .endprolog
    cvtpd2pi mm2, XMMWORD PTR [rdx+16]
    movq QWORD PTR [r8+8], mm2
    emms
    ret
op_cvtpd2pi_mem_disp ENDP

op_cvttpd2pi_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    cvtpd2pi mm3, xmm1
ELSE
    cvttpd2pi mm3, xmm1
ENDIF
    movq QWORD PTR [r8], mm3
    emms
    ret
op_cvttpd2pi_reg ENDP

op_cvttpd2pi_mem PROC FRAME
    .endprolog
    cvttpd2pi mm4, XMMWORD PTR [rdx]
    movq QWORD PTR [r8], mm4
    emms
    ret
op_cvttpd2pi_mem ENDP

op_movdq2q PROC FRAME
    .endprolog
    movdqa xmm2, XMMWORD PTR [rcx]
    movdq2q mm5, xmm2
    movq QWORD PTR [r8], mm5
    emms
    ret
op_movdq2q ENDP

op_movq2dq PROC FRAME
    .endprolog
    movdqa xmm3, XMMWORD PTR [rcx]
    movq mm6, QWORD PTR [rdx]
    movq2dq xmm3, mm6
    movdqa XMMWORD PTR [r8], xmm3
    emms
    ret
op_movq2dq ENDP

op_emms_standalone PROC FRAME
    .endprolog
    movdqu xmm0, XMMWORD PTR [rcx+1]
    emms
    movdqu XMMWORD PTR [r8+1], xmm0
    ret
op_emms_standalone ENDP

op_movq_reg_mem PROC FRAME
    .endprolog
    movq mm0, QWORD PTR [rcx]
    movq mm1, mm0
    movq rax, mm1
    movq mm2, rax
    movq QWORD PTR [r8], mm2
    emms
    ret
op_movq_reg_mem ENDP

op_movdqa_reg_ls PROC FRAME
    .endprolog
    movdqa xmm4, XMMWORD PTR [rcx]
    movdqa xmm5, xmm4
    movdqa XMMWORD PTR [r8], xmm5
    ret
op_movdqa_reg_ls ENDP

op_movdqu_reg_ls PROC FRAME
    .endprolog
    movdqu xmm0, XMMWORD PTR [rcx+3]
    movdqu xmm1, xmm0
    movdqu XMMWORD PTR [r8+3], xmm1
    ret
op_movdqu_reg_ls ENDP

op_mmx_sse2_bridge PROC FRAME
    .endprolog
    movdqu xmm0, XMMWORD PTR [rcx]
    cvtpd2pi mm0, xmm0
    movq2dq xmm1, mm0
    movdqa XMMWORD PTR [r8], xmm1
    emms
    movdqa xmm2, XMMWORD PTR [rdx]
    cvttpd2pi mm7, xmm2
    movq r9, mm7
    movq mm3, r9
    movq QWORD PTR [r8+16], mm3
    movdq2q mm6, xmm1
    movq QWORD PTR [r8+24], mm6
    emms
    ret
op_mmx_sse2_bridge ENDP

op_cvtpi2pd_reg PROC FRAME
    .endprolog
    movq mm4, QWORD PTR [rcx]
IFDEF ALTER_IMM
    movq mm5, QWORD PTR [rdx]
    cvtpi2pd xmm3, mm5
ELSE
    cvtpi2pd xmm3, mm4
ENDIF
    movdqa XMMWORD PTR [r8], xmm3
    emms
    ret
op_cvtpi2pd_reg ENDP

op_cvtpi2pd_mem PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    cvtpi2pd xmm1, QWORD PTR [rdx]
ELSE
    cvtpi2pd xmm1, QWORD PTR [rcx]
ENDIF
    movdqa XMMWORD PTR [r8], xmm1
    ret
op_cvtpi2pd_mem ENDP

op_cvtpi2pd_mem_disp PROC FRAME
    .endprolog
    cvtpi2pd xmm2, QWORD PTR [rdx+1]
    movdqu XMMWORD PTR [r8+3], xmm2
    ret
op_cvtpi2pd_mem_disp ENDP

; The CVTPI2PD source must be read even though xmm1 is dead after this instruction.
op_cvtpi2pd_discarded_read PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa XMMWORD PTR [r8], xmm0
    cvtpi2pd xmm1, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8+16], xmm0
    ret
op_cvtpi2pd_discarded_read ENDP

op_cvtpi2pd_bridge PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdq2q mm2, xmm0
    cvtpi2pd xmm4, mm2
    movdqa XMMWORD PTR [r8], xmm4
    cvtpd2pi mm6, xmm4
    movq rax, mm6
    movq mm3, rax
    cvtpi2pd xmm5, mm3
    movdqa XMMWORD PTR [r8+16], xmm5
    emms
    ret
op_cvtpi2pd_bridge ENDP

END
