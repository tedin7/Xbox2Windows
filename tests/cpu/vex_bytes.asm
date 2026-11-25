; coverage: record cpu_aot_vex_bytes
; VEX.128 byte-domain packed-integer leaf functions (vpandn, vpavgb, vpmaddubsw, vpshufb),
; an altered build for differential controls, and their lifted C++.
; Each takes RCX and RDX (addresses of two 16-byte inputs) and R8 (address of a 16-byte output).
.code

pandn_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTES
    ; Swap operands to expose (~a) & b asymmetry.
    vpandn xmm2, xmm1, xmm0
ELSE
    vpandn xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pandn_reg ENDP

pandn_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpandn xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pandn_mem ENDP

pavgb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTES
    ; Use 16-bit word average instead of 8-bit byte average.
    vpavgw xmm2, xmm0, xmm1
ELSE
    vpavgb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pavgb_reg ENDP

pavgb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpavgb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pavgb_mem ENDP

pmaddubsw_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTES
    ; Swap unsigned/signed sources to expose operand asymmetry.
    vpmaddubsw xmm2, xmm1, xmm0
ELSE
    vpmaddubsw xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pmaddubsw_reg ENDP

pmaddubsw_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpmaddubsw xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pmaddubsw_mem ENDP

pshufb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTES
    ; Swap data and shuffle-control registers.
    vpshufb xmm2, xmm1, xmm0
ELSE
    vpshufb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pshufb_reg ENDP

pshufb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpshufb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pshufb_mem ENDP

; Chained operations exercising XMM0..XMM5, [rcx]/[rdx]/[r8] memory sources, and store-reload order.
bytes_mem_r8_chain PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpmaddubsw xmm2, xmm0, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm2
    vpandn xmm3, xmm2, xmmword ptr [rcx]
    vpavgb xmm4, xmm3, xmmword ptr [r8]
    vpshufb xmm5, xmm4, xmmword ptr [rdx]
    vpandn xmm0, xmm5, xmmword ptr [rcx]
    vpmaddubsw xmm1, xmm0, xmmword ptr [r8]
    vpshufb xmm2, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm2
    vzeroupper
    ret
bytes_mem_r8_chain ENDP

; Aligned-move variant: buffers must be 16-byte aligned.
bytes_aligned PROC FRAME
    .endprolog
    vmovdqa xmm0, xmmword ptr [rcx]
    vpandn xmm1, xmm0, xmmword ptr [rdx]
    vpavgb xmm2, xmm1, xmm0
    vpmaddubsw xmm3, xmm2, xmmword ptr [rdx]
    vpshufb xmm4, xmm3, xmm1
    vmovdqa xmmword ptr [r8], xmm4
    ret
bytes_aligned ENDP

END
