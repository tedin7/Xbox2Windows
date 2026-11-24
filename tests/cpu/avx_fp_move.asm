; coverage: source cpu_aot_avx_fp_move
; Source-built MASM fixture exercising 4 AVX forms from docs/coverage/x64_zen2.tsv
; group AVX: vmovq, vmaskmovdqu, vaddsubps and vaddsubpd, along with VEX transport glue
; (vmovups, vmovaps, vmovdqu, vmovdqa, vzeroupper, 64-bit mov).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 32-byte aligned)
;   RDX = const void* input_b (32 bytes, 32-byte aligned)
;   R8  = void*       output  (32 bytes, 32-byte aligned)
.code

; --- vmovq: XMM<-XMM, XMM<-m64, m64<-XMM, XMM<-r64, r64<-XMM (zeroing [255:64]) ---

op_vmovq_xmm_xmm PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_OP
    vmovq xmm0, xmm0
ELSE
    vmovq xmm0, xmm1
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vmovq_xmm_xmm ENDP

op_vmovq_xmm_m64 PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
IFDEF ALTER_OP
    vmovq xmm0, QWORD PTR [rdx+9]
ELSE
    vmovq xmm0, QWORD PTR [rdx+1]
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vmovq_xmm_m64 ENDP

op_vmovq_m64_xmm PROC FRAME
    .endprolog
    vmovups xmm0, XMMWORD PTR [rcx]
    vmovq QWORD PTR [r8+9], xmm0
    ret
op_vmovq_m64_xmm ENDP

op_vmovq_xmm_r64 PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    mov rax, QWORD PTR [rdx]
    vmovq xmm0, rax
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vmovq_xmm_r64 ENDP

op_vmovq_r64_xmm PROC FRAME
    .endprolog
    vmovups xmm0, XMMWORD PTR [rcx]
    vmovq r9, xmm0
    mov QWORD PTR [r8+8], r9
    ret
op_vmovq_r64_xmm ENDP

; --- vmaskmovdqu: conditional byte store to [rdi] with rdi save/restore ---

op_vmaskmovdqu PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
    mov r10, rdi
    mov rdi, r8
IFDEF ALTER_OP
    vmaskmovdqu xmm1, xmm0
ELSE
    vmaskmovdqu xmm0, xmm1
ENDIF
    mov rdi, r10
    ret
op_vmaskmovdqu ENDP

; --- vaddsubps / vaddsubpd: even elements src1 - src2, odd elements src1 + src2 ---
; The VEX.128 forms must zero [255:128] of the destination (the destination is preloaded
; with nonzero upper bytes). Memory forms read [rdx] with an unaligned offset for m128.
; ALTER_OP swaps the register sources (or loads src1 from [rdx]) to reverse the subtraction.

op_vaddsubps_xmm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_OP
    vaddsubps xmm2, xmm1, xmm0
ELSE
    vaddsubps xmm2, xmm0, xmm1
ENDIF
    vmovups YMMWORD PTR [r8], ymm2
    ret
op_vaddsubps_xmm_reg ENDP

op_vaddsubps_xmm_mem PROC FRAME
    .endprolog
IFDEF ALTER_OP
    vmovups ymm0, YMMWORD PTR [rdx]
ELSE
    vmovups ymm0, YMMWORD PTR [rcx]
ENDIF
    vaddsubps xmm0, xmm0, XMMWORD PTR [rdx+4]
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vaddsubps_xmm_mem ENDP

op_vaddsubps_ymm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
IFDEF ALTER_OP
    vaddsubps ymm3, ymm1, ymm0
ELSE
    vaddsubps ymm3, ymm0, ymm1
ENDIF
    vmovups YMMWORD PTR [r8], ymm3
    ret
op_vaddsubps_ymm_reg ENDP

op_vaddsubps_ymm_mem PROC FRAME
    .endprolog
IFDEF ALTER_OP
    vmovups ymm0, YMMWORD PTR [rdx]
ELSE
    vmovups ymm0, YMMWORD PTR [rcx]
ENDIF
    vaddsubps ymm0, ymm0, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vaddsubps_ymm_mem ENDP

op_vaddsubpd_xmm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_OP
    vaddsubpd xmm2, xmm1, xmm0
ELSE
    vaddsubpd xmm2, xmm0, xmm1
ENDIF
    vmovups YMMWORD PTR [r8], ymm2
    ret
op_vaddsubpd_xmm_reg ENDP

op_vaddsubpd_xmm_mem PROC FRAME
    .endprolog
IFDEF ALTER_OP
    vmovups ymm0, YMMWORD PTR [rdx]
ELSE
    vmovups ymm0, YMMWORD PTR [rcx]
ENDIF
    vaddsubpd xmm0, xmm0, XMMWORD PTR [rdx+8]
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vaddsubpd_xmm_mem ENDP

op_vaddsubpd_ymm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
IFDEF ALTER_OP
    vaddsubpd ymm3, ymm1, ymm0
ELSE
    vaddsubpd ymm3, ymm0, ymm1
ENDIF
    vmovups YMMWORD PTR [r8], ymm3
    ret
op_vaddsubpd_ymm_reg ENDP

op_vaddsubpd_ymm_mem PROC FRAME
    .endprolog
IFDEF ALTER_OP
    vmovups ymm0, YMMWORD PTR [rdx]
ELSE
    vmovups ymm0, YMMWORD PTR [rcx]
ENDIF
    vaddsubpd ymm0, ymm0, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    ret
op_vaddsubpd_ymm_mem ENDP

; --- Chained vzeroupper pipeline and dead-read fault-ordering controls ---

op_pipeline PROC FRAME
    .endprolog
    vmovaps ymm0, YMMWORD PTR [rcx]
    vmovdqa ymm1, YMMWORD PTR [rdx]
    vzeroupper
    vmovq xmm2, QWORD PTR [rdx+16]
    vmovq r9, xmm2
    vmovq xmm3, r9
    vmovdqu XMMWORD PTR [r8+16], xmm3
    vmovups XMMWORD PTR [r8], xmm0
    mov r11, rdi
    mov rdi, r8
    vmaskmovdqu xmm0, xmm1
    mov rdi, r11
    vmovq QWORD PTR [r8+24], xmm2
    ret
op_pipeline ENDP

; The discarded reads must still fault after the first store and before the second.
op_discarded_read PROC FRAME
    .endprolog
    vmovups xmm0, XMMWORD PTR [rcx]
    vmovups XMMWORD PTR [r8], xmm0
    vmovups ymm5, YMMWORD PTR [rdx]
    vmovups XMMWORD PTR [r8+16], xmm0
    ret
op_discarded_read ENDP

op_discarded_read64 PROC FRAME
    .endprolog
    vmovups xmm0, XMMWORD PTR [rcx]
    vmovups XMMWORD PTR [r8], xmm0
    vmovq xmm5, QWORD PTR [rdx]
    vmovups XMMWORD PTR [r8+16], xmm0
    ret
op_discarded_read64 ENDP

END
