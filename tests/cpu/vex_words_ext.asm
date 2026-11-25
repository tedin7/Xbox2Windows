; coverage: source cpu_aot_vex_words_ext
; VEX.128 16-bit packed-word leaf functions (vpabsw, vpsignw, vpmulhrsw, vphaddw), an altered
; build for negative differential controls, and their lifted C++.
; Each takes RCX and RDX (addresses of two 16-byte inputs) and R8 (address of a 16-byte output).
.code

pabsw_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpabsw xmm1, xmm0
    vpabsw xmm2, xmm1
    vmovdqu xmmword ptr [r8], xmm2
    ret
pabsw_reg ENDP

pabsw_mem PROC FRAME
    .endprolog
    vpabsw xmm4, xmmword ptr [rcx]
    vmovdqu xmmword ptr [r8], xmm4
    vpabsw xmm5, xmmword ptr [r8]
    vmovdqu xmmword ptr [r8], xmm5
    ret
pabsw_mem ENDP

signw_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_EXT
    ; Zero-mask bug control: map zero sign words to +1 so zero masks return input_a unchanged.
    vpxor xmm4, xmm4, xmm4
    vpcmpeqw xmm3, xmm1, xmm4
    vpsrlw xmm3, xmm3, 15
    vpor xmm3, xmm1, xmm3
    vpsignw xmm2, xmm0, xmm3
ELSE
    vpsignw xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
signw_reg ENDP

signw_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpsignw xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
signw_mem ENDP

mulhrsw_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_EXT
    ; Drop rounding (+0x4000) and scale via vpmulhw as an altered-build differential control.
    vpmulhw xmm2, xmm0, xmm1
ELSE
    vpmulhrsw xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
mulhrsw_reg ENDP

mulhrsw_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpmulhrsw xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
mulhrsw_mem ENDP

phaddw_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_EXT
    ; Swap src1 and src2 halves as an altered-build differential control.
    vphaddw xmm2, xmm1, xmm0
ELSE
    vphaddw xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
phaddw_reg ENDP

phaddw_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vphaddw xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
phaddw_mem ENDP

; Chained operations exercising XMM0..XMM5, [rcx]/[rdx]/[r8] memory sources, and store-reload order.
ext_mem_r8_chain PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpmulhrsw xmm2, xmm0, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm2
    vpabsw xmm3, xmmword ptr [r8]
    vpsignw xmm4, xmm3, xmmword ptr [rcx]
    vphaddw xmm5, xmm4, xmmword ptr [r8]
    vpmulhrsw xmm0, xmm5, xmmword ptr [rcx]
    vphaddw xmm1, xmm0, xmmword ptr [rdx]
    vpsignw xmm2, xmm1, xmmword ptr [r8]
    vmovdqu xmmword ptr [r8], xmm2
    vzeroupper
    ret
ext_mem_r8_chain ENDP

; Aligned-move variant: buffers must be 16-byte aligned.
ext_aligned PROC FRAME
    .endprolog
    vmovdqa xmm0, xmmword ptr [rcx]
    vpabsw xmm1, xmmword ptr [rdx]
    vpmulhrsw xmm2, xmm0, xmm1
    vpsignw xmm3, xmm2, xmm0
    vphaddw xmm4, xmm3, xmm1
    vmovdqa xmmword ptr [r8], xmm4
    ret
ext_aligned ENDP

END
