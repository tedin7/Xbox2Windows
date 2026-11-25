; coverage: record cpu_aot_vex_bytes_arith
; VEX.128 byte-domain arithmetic and compare leaf functions (vpaddb, vpsubb, vpcmpeqb, vpcmpgtb),
; an altered build for differential controls, and their lifted C++.
; Each takes RCX and RDX (addresses of two 16-byte inputs) and R8 (address of a 16-byte output).
.code

paddb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTE_ARITH
    vpsubb xmm2, xmm0, xmm1
ELSE
    vpaddb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
paddb_reg ENDP

paddb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpaddb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
paddb_mem ENDP

psubb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTE_ARITH
    ; Swap operands to expose (a - b) asymmetry.
    vpsubb xmm2, xmm1, xmm0
ELSE
    vpsubb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
psubb_reg ENDP

psubb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpsubb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
psubb_mem ENDP

; dst == src1 == src2 zeroing on defined register.
psubb_self PROC FRAME
    .endprolog
    vmovdqu xmm2, xmmword ptr [rcx]
    vpsubb xmm2, xmm2, xmm2
    vmovdqu xmmword ptr [r8], xmm2
    ret
psubb_self ENDP

pcmpeqb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTE_ARITH
    vpcmpgtb xmm2, xmm0, xmm1
ELSE
    vpcmpeqb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pcmpeqb_reg ENDP

pcmpeqb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpcmpeqb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pcmpeqb_mem ENDP

pcmpgtb_reg PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
IFDEF ALTER_BYTE_ARITH
    ; Swap operands to expose signed (a > b) asymmetry.
    vpcmpgtb xmm2, xmm1, xmm0
ELSE
    vpcmpgtb xmm2, xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm2
    ret
pcmpgtb_reg ENDP

pcmpgtb_mem PROC FRAME
    .endprolog
    vmovdqu xmm3, xmmword ptr [rcx]
    vpcmpgtb xmm4, xmm3, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm4
    ret
pcmpgtb_mem ENDP

; Chained operations exercising XMM0..XMM5, [rcx]/[rdx]/[r8] memory sources, and store-reload order.
bytes_arith_mem_r8_chain PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpaddb xmm2, xmm0, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm2
    vpsubb xmm3, xmm2, xmmword ptr [rcx]
    vpcmpeqb xmm4, xmm3, xmmword ptr [r8]
    vpcmpgtb xmm5, xmm4, xmmword ptr [rdx]
    vpaddb xmm0, xmm5, xmmword ptr [rcx]
    vpsubb xmm1, xmm0, xmmword ptr [r8]
    vpcmpgtb xmm2, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm2
    vzeroupper
    ret
bytes_arith_mem_r8_chain ENDP

; Aligned-move variant: buffers must be 16-byte aligned.
bytes_arith_aligned PROC FRAME
    .endprolog
    vmovdqa xmm0, xmmword ptr [rcx]
    vpaddb xmm1, xmm0, xmmword ptr [rdx]
    vpsubb xmm2, xmm1, xmm0
    vpcmpeqb xmm3, xmm2, xmmword ptr [rdx]
    vpcmpgtb xmm4, xmm3, xmm1
    vmovdqa xmmword ptr [r8], xmm4
    ret
bytes_arith_aligned ENDP

END
