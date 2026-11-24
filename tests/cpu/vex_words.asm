; coverage: source cpu_aot_vex_words
; VEX.128 16-bit packed-integer leaf functions. Each takes RCX and RDX, the addresses of two
; 16-byte inputs, and R8, the address of a 16-byte output, and returns nothing.
.code

; Widen the bytes of both inputs to words, combine them and pack back with unsigned saturation.
widen_mix PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpxor xmm2, xmm2, xmm2
    vpunpcklbw xmm3, xmm0, xmm2
    vpunpckhbw xmm4, xmm0, xmm2
    vpunpcklbw xmm5, xmm1, xmm2
    vpunpckhbw xmm1, xmm1, xmm2
    vpaddw xmm3, xmm3, xmm5
    vpsubw xmm4, xmm4, xmm1
    vpsraw xmm3, xmm3, 1
    vpsllw xmm4, xmm4, 2
    vpackuswb xmm0, xmm3, xmm4
    vmovdqu xmmword ptr [r8], xmm0
    vzeroupper
    ret
widen_mix ENDP

; Signed word arithmetic with memory sources, then interleave and combine the results.
word_ops PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpmulhw xmm1, xmm0, xmmword ptr [rdx]
    vpavgw xmm2, xmm0, xmmword ptr [rdx]
    vpmaxsw xmm3, xmm1, xmm2
    vpminsw xmm4, xmm1, xmm2
    vpunpcklwd xmm5, xmm3, xmm4
    vpunpckhwd xmm2, xmm3, xmm4
    vpxor xmm5, xmm5, xmm2
    vpor xmm5, xmm5, xmm1
    vpand xmm5, xmm5, xmm0
    vpaddw xmm5, xmm5, xmm3
    vmovdqu xmmword ptr [r8], xmm5
    ret
word_ops ENDP

; Shift counts at and past the word width: arithmetic shifts fill with the sign, logical shifts
; clear the lane. Only the volatile XMM0-XMM5 are used.
shift_edges PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm5, xmmword ptr [rdx]
IFDEF LOGICAL_SHIFT
    vpsrlw xmm1, xmm0, 15
ELSE
    vpsraw xmm1, xmm0, 15
ENDIF
    vpsraw xmm2, xmm5, 16
    vpsllw xmm3, xmm0, 16
    vpsllw xmm4, xmm5, 1
    vpaddw xmm1, xmm1, xmm2
    vpor xmm1, xmm1, xmm3
    vpxor xmm1, xmm1, xmm4
    vpsrlw xmm2, xmm0, 0
    vpaddw xmm1, xmm1, xmm2
    vpsrlw xmm3, xmm5, 16
    vpor xmm1, xmm1, xmm3
    vmovdqu xmmword ptr [r8], xmm1
    ret
shift_edges ENDP

; Store, then read the inputs again and store a second result. When the output aliases an
; input, the second reads see the first store, so the order of reads and writes is observable.
store_reload PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpaddw xmm0, xmm0, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm0
    vpsubw xmm1, xmm0, xmmword ptr [rcx]
    vpmaxsw xmm1, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm1
    ret
store_reload ENDP

; Aligned forms: every buffer must be 16-byte aligned, or vmovdqa faults.
aligned_ops PROC FRAME
    .endprolog
    vmovdqa xmm0, xmmword ptr [rcx]
    vmovdqa xmm1, xmmword ptr [rdx]
    vmovdqa xmm2, xmm0
    vpavgw xmm2, xmm2, xmm1
    vpunpckhwd xmm3, xmm2, xmm1
    vmovdqa xmmword ptr [r8], xmm3
    ret
aligned_ops ENDP

; Pack signed words made from both inputs, then reload the packed bytes from the output.
byte_pack_reload PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vpunpcklbw xmm2, xmm0, xmm1
    vpunpckhbw xmm3, xmm0, xmm1
    vpsraw xmm3, xmm3, 1
    vpackuswb xmm2, xmm2, xmm3
    vmovdqu xmmword ptr [r8], xmm2
    vpsrlw xmm0, xmm0, 32
    vpxor xmm0, xmm0, xmmword ptr [r8]
    vmovdqu xmmword ptr [r8], xmm0
    ret
byte_pack_reload ENDP

; Saturating add and subtract, signed and unsigned, with memory sources, folded into one result.
sat_words PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vpaddsw xmm1, xmm0, xmmword ptr [rdx]
    vpaddusw xmm2, xmm0, xmmword ptr [rdx]
    vpsubsw xmm3, xmm0, xmmword ptr [rdx]
    vpsubusw xmm4, xmm0, xmmword ptr [rdx]
    vpaddw xmm2, xmm2, xmm3
    vpxor xmm1, xmm1, xmm2
    vpxor xmm1, xmm1, xmm4
    vmovdqu xmmword ptr [r8], xmm1
    ret
sat_words ENDP

; Low and high word products and unsigned word extremes. In the altered build the unsigned
; high product becomes the signed one, so lanes with bit 15 set in either input change.
mul_minmax_words PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm5, xmmword ptr [rdx]
    vpmullw xmm1, xmm0, xmm5
IFDEF LOGICAL_SHIFT
    vpmulhw xmm2, xmm0, xmm5
ELSE
    vpmulhuw xmm2, xmm0, xmm5
ENDIF
    vpmaxuw xmm3, xmm0, xmm5
    vpminuw xmm4, xmm0, xmmword ptr [rdx]
    vpaddw xmm1, xmm1, xmm2
    vpxor xmm1, xmm1, xmm3
    vpsubw xmm1, xmm1, xmm4
    vmovdqu xmmword ptr [r8], xmm1
    ret
mul_minmax_words ENDP

; Word compares. Each lane ends as exactly one code: equal 1, A above B 3, A below B 7, with the
; signed order. The compares are all ones or all zeros before the shifts.
cmp_words PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vmovdqu xmm5, xmmword ptr [rdx]
    vpcmpeqw xmm1, xmm0, xmm5
    vpcmpgtw xmm2, xmm0, xmmword ptr [rdx]
    vpcmpgtw xmm3, xmm5, xmm0
    vpsrlw xmm1, xmm1, 15
    vpsrlw xmm2, xmm2, 14
    vpsrlw xmm3, xmm3, 13
    vpor xmm1, xmm1, xmm2
    vpor xmm1, xmm1, xmm3
    vmovdqu xmmword ptr [r8], xmm1
    ret
cmp_words ENDP
END
