; coverage: record cpu_aot_sha1
; Source-owned differential fixture for Intel SHA-1 NI instructions:
; SHA1RNDS4, SHA1NEXTE, SHA1MSG1, and SHA1MSG2 across register and flat [rdx]/[r8]
; memory operands, all four SHA1RNDS4 imm8 functions (plus upper-bit masking),
; self-operand forms, and a multi-instruction pipeline on volatile XMM0..XMM5.
;   RCX = const void* input_a (16 bytes; 16B-aligned for movdqa / sha1* m128)
;   RDX = const void* input_b (16 bytes; 16B-aligned for movdqa / sha1* m128)
;   R8  = void*       output  (16 bytes; 16B-aligned for movdqa / sha1* m128)
.code

op_sha1rnds4_0_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1rnds4 xmm0, xmm1, 1
ELSE
    sha1rnds4 xmm0, xmm1, 0
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_0_reg ENDP

op_sha1rnds4_1_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1rnds4 xmm0, xmm1, 2
ELSE
    sha1rnds4 xmm0, xmm1, 1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_1_reg ENDP

op_sha1rnds4_2_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1rnds4 xmm0, xmm1, 3
ELSE
    sha1rnds4 xmm0, xmm1, 2
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_2_reg ENDP

op_sha1rnds4_3_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1rnds4 xmm0, xmm1, 0
ELSE
    sha1rnds4 xmm0, xmm1, 3
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_3_reg ENDP

; Upper bits [7:2] of SHA1RNDS4 imm8 must be ignored (0FCh->0, 0FDh->1, 0FEh->2, 0FFh->3).
op_sha1rnds4_hi_imm PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    sha1rnds4 xmm0, xmm1, 0FCh
    sha1rnds4 xmm0, xmm1, 0FDh
    sha1rnds4 xmm0, xmm1, 0FEh
    sha1rnds4 xmm0, xmm1, 0FFh
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_hi_imm ENDP

op_sha1rnds4_0_mem PROC FRAME
    .endprolog
    movdqa xmm2, xmmword ptr [rcx]
    sha1rnds4 xmm2, xmmword ptr [rdx], 0
    movdqa xmmword ptr [r8], xmm2
    ret
op_sha1rnds4_0_mem ENDP

op_sha1rnds4_1_mem PROC FRAME
    .endprolog
    movdqa xmm3, xmmword ptr [rcx]
    sha1rnds4 xmm3, xmmword ptr [rdx], 1
    movdqa xmmword ptr [r8], xmm3
    ret
op_sha1rnds4_1_mem ENDP

op_sha1rnds4_2_mem PROC FRAME
    .endprolog
    movdqa xmm4, xmmword ptr [rcx]
    sha1rnds4 xmm4, xmmword ptr [rdx], 2
    movdqa xmmword ptr [r8], xmm4
    ret
op_sha1rnds4_2_mem ENDP

op_sha1rnds4_3_mem PROC FRAME
    .endprolog
    movdqa xmm5, xmmword ptr [rcx]
    sha1rnds4 xmm5, xmmword ptr [rdx], 3
    movdqa xmmword ptr [r8], xmm5
    ret
op_sha1rnds4_3_mem ENDP

op_sha1rnds4_self PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    sha1rnds4 xmm0, xmm0, 2
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1rnds4_self ENDP

op_sha1nexte_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1nexte xmm1, xmm0
    movdqu xmm0, xmm1
ELSE
    sha1nexte xmm0, xmm1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1nexte_reg ENDP

op_sha1nexte_mem PROC FRAME
    .endprolog
    movdqa xmm2, xmmword ptr [rcx]
    sha1nexte xmm2, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm2
    ret
op_sha1nexte_mem ENDP

op_sha1nexte_self PROC FRAME
    .endprolog
    movdqu xmm3, xmmword ptr [rcx]
    sha1nexte xmm3, xmm3
    movdqu xmmword ptr [r8], xmm3
    ret
op_sha1nexte_self ENDP

op_sha1msg1_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1msg2 xmm0, xmm1
ELSE
    sha1msg1 xmm0, xmm1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1msg1_reg ENDP

op_sha1msg1_mem PROC FRAME
    .endprolog
    movdqa xmm4, xmmword ptr [rcx]
    sha1msg1 xmm4, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm4
    ret
op_sha1msg1_mem ENDP

op_sha1msg1_self PROC FRAME
    .endprolog
    movdqu xmm5, xmmword ptr [rcx]
    sha1msg1 xmm5, xmm5
    movdqu xmmword ptr [r8], xmm5
    ret
op_sha1msg1_self ENDP

op_sha1msg2_reg PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_SHA1
    sha1msg1 xmm0, xmm1
ELSE
    sha1msg2 xmm0, xmm1
ENDIF
    movdqu xmmword ptr [r8], xmm0
    ret
op_sha1msg2_reg ENDP

op_sha1msg2_mem PROC FRAME
    .endprolog
    movdqa xmm2, xmmword ptr [rcx]
    sha1msg2 xmm2, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm2
    ret
op_sha1msg2_mem ENDP

op_sha1msg2_self PROC FRAME
    .endprolog
    movdqu xmm4, xmmword ptr [rcx]
    sha1msg2 xmm4, xmm4
    movdqu xmmword ptr [r8], xmm4
    ret
op_sha1msg2_self ENDP

; Multi-instruction pipeline exercising XMM0..XMM5, movdqa/movdqu reg copies,
; and an [r8] store followed by a flat [r8] memory source in sha1nexte.
op_sha1_pipeline PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
    movdqa xmm2, xmm0
    sha1msg1 xmm2, xmm1
    sha1msg2 xmm2, xmm1
    movdqa xmmword ptr [r8], xmm2
    movdqu xmm3, xmm0
    sha1rnds4 xmm3, xmm1, 0
    movdqa xmm4, xmm3
    sha1nexte xmm4, xmmword ptr [r8]
    movdqa xmm5, xmm3
    sha1rnds4 xmm5, xmm4, 1
    movdqa xmmword ptr [r8], xmm5
    ret
op_sha1_pipeline ENDP

END
