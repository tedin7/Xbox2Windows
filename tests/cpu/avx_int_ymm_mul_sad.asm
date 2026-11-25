; coverage: source cpu_aot_avx_int_ymm_mul_sad
; Source-owned differential fixture for 256-bit (VEX.L=1, AVX2) packed integer
; multiply and sum-of-absolute-differences forms: VPMULDQ, VPMULUDQ, VPMULHRSW, VPSADBW.
; Each 128-bit half is processed independently.
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       output  (32 bytes)
.code

op_vpmuldq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpmuludq ymm0, ymm1, ymm2
ELSE
    vpmuldq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmuldq_ymm ENDP

op_vpmuludq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpmuldq ymm0, ymm1, ymm2
ELSE
    vpmuludq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmuludq_ymm ENDP

op_vpmulhrsw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpsadbw ymm0, ymm1, ymm2
ELSE
    vpmulhrsw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmulhrsw_ymm ENDP

op_vpsadbw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpmulhrsw ymm0, ymm1, ymm2
ELSE
    vpsadbw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpsadbw_ymm ENDP

op_vpmuldq_ymm_dst_src1 PROC FRAME
    .endprolog
    vmovdqu ymm3, YMMWORD PTR [rcx]
    vmovdqu ymm4, YMMWORD PTR [rdx]
    vpmuldq ymm3, ymm3, ymm4
    vmovdqu YMMWORD PTR [r8], ymm3
    vzeroupper
    ret
op_vpmuldq_ymm_dst_src1 ENDP

op_vpmuludq_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vmovdqu ymm5, YMMWORD PTR [rdx]
    vpmuludq ymm5, ymm4, ymm5
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_vpmuludq_ymm_dst_src2 ENDP

op_vpmulhrsw_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vpmulhrsw ymm2, ymm2, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpmulhrsw_ymm_all ENDP

op_vpsadbw_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpsadbw ymm2, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpsadbw_ymm_dst_src2 ENDP

op_vpmuldq_ymm_then_xmm_vpmuludq PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpmuldq ymm0, ymm1, ymm2
    vpmuludq xmm0, xmm0, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmuldq_ymm_then_xmm_vpmuludq ENDP

op_chain_ymm_mul_sad PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vpmuldq ymm2, ymm0, ymm1
    vpmuludq ymm3, ymm1, ymm0
    vpmulhrsw ymm4, ymm2, ymm3
    vpsadbw ymm5, ymm4, ymm0
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_chain_ymm_mul_sad ENDP

END
