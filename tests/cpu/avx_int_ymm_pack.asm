; coverage: source cpu_aot_avx_int_ymm_pack
; Source-owned differential fixture for 256-bit (VEX.L=1, AVX2) packed-integer
; pack-with-saturation forms: VPACKSSWB, VPACKSSDW, VPACKUSDW.
; Each 128-bit half is processed independently.
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       output  (32 bytes)
.code

op_vpacksswb_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpackssdw ymm0, ymm1, ymm2
ELSE
    vpacksswb ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpacksswb_ymm ENDP

op_vpackssdw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpackusdw ymm0, ymm1, ymm2
ELSE
    vpackssdw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpackssdw_ymm ENDP

op_vpackusdw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpackssdw ymm0, ymm1, ymm2
ELSE
    vpackusdw ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpackusdw_ymm ENDP

op_vpacksswb_ymm_dst_src1 PROC FRAME
    .endprolog
    vmovdqu ymm3, YMMWORD PTR [rcx]
    vmovdqu ymm4, YMMWORD PTR [rdx]
    vpacksswb ymm3, ymm3, ymm4
    vmovdqu YMMWORD PTR [r8], ymm3
    vzeroupper
    ret
op_vpacksswb_ymm_dst_src1 ENDP

op_vpackssdw_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vmovdqu ymm5, YMMWORD PTR [rdx]
    vpackssdw ymm5, ymm4, ymm5
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_vpackssdw_ymm_dst_src2 ENDP

op_vpackusdw_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vpackusdw ymm2, ymm2, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpackusdw_ymm_all ENDP

op_vpacksswb_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpacksswb ymm2, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpacksswb_ymm_dst_src2 ENDP

op_vpackusdw_ymm_then_xmm_vpacksswb PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpackusdw ymm0, ymm1, ymm2
    vpacksswb xmm0, xmm0, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpackusdw_ymm_then_xmm_vpacksswb ENDP

op_chain_ymm_pack PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vpackssdw ymm2, ymm0, ymm1
    vpackusdw ymm3, ymm1, ymm0
    vpacksswb ymm0, ymm2, ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_chain_ymm_pack ENDP

END
