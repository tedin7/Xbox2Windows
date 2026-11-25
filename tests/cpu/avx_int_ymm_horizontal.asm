; coverage: source cpu_aot_avx_int_ymm_horizontal
; Source-owned differential fixture for 256-bit (VEX.L=1, AVX2) horizontal packed-integer
; add and subtract forms: VPHADDW, VPHADDD, VPHADDSW, VPHSUBW, VPHSUBD, VPHSUBSW.
; Each 128-bit half is processed independently.
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       output  (32 bytes)
.code

op_vphaddw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphsubw ymm0, ymm1, ymm2
ELSE
    vphaddw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphaddw_ymm ENDP

op_vphaddd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphsubd ymm0, ymm1, ymm2
ELSE
    vphaddd ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphaddd_ymm ENDP

op_vphaddsw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphsubsw ymm0, ymm1, ymm2
ELSE
    vphaddsw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphaddsw_ymm ENDP

op_vphsubw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphaddw ymm0, ymm1, ymm2
ELSE
    vphsubw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphsubw_ymm ENDP

op_vphsubd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphaddd ymm0, ymm1, ymm2
ELSE
    vphsubd ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphsubd_ymm ENDP

op_vphsubsw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vphaddsw ymm0, ymm1, ymm2
ELSE
    vphsubsw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphsubsw_ymm ENDP

op_vphaddw_ymm_dst_src1 PROC FRAME
    .endprolog
    vmovdqu ymm3, YMMWORD PTR [rcx]
    vmovdqu ymm4, YMMWORD PTR [rdx]
    vphaddw ymm3, ymm3, ymm4
    vmovdqu YMMWORD PTR [r8], ymm3
    vzeroupper
    ret
op_vphaddw_ymm_dst_src1 ENDP

op_vphaddd_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vmovdqu ymm5, YMMWORD PTR [rdx]
    vphaddd ymm5, ymm4, ymm5
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_vphaddd_ymm_dst_src2 ENDP

op_vphaddsw_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vphaddsw ymm2, ymm2, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vphaddsw_ymm_all ENDP

op_vphsubw_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vphsubw ymm2, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vphsubw_ymm_dst_src2 ENDP

op_vphaddsw_ymm_then_xmm_vphsubsw PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vphaddsw ymm0, ymm1, ymm2
    vphsubsw xmm0, xmm0, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vphaddsw_ymm_then_xmm_vphsubsw ENDP

op_chain_ymm_horizontal PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vphaddw ymm2, ymm0, ymm1
    vphsubw ymm3, ymm1, ymm0
    vphaddsw ymm4, ymm2, ymm3
    vphsubsw ymm5, ymm4, ymm0
    vphaddd ymm2, ymm5, ymm3
    vphsubd ymm0, ymm2, ymm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_chain_ymm_horizontal ENDP

END
