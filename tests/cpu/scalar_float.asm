; coverage: source cpu_aot_scalar_float
; Windows x64 leaves built from scalar SSE/SSE2 instructions. An argument arrives in the XMM
; register of its slot when it is floating point and in the integer register otherwise.
; SWAP_OPERANDS builds the control: one operand order or source differs in every function.
.code
; float blend_ss(float a, float b, float* c, float d)
; c[1] = (a + b) * c[0] - d, and the result is a / c[1].
blend_ss PROC FRAME
    .endprolog
    movaps xmm2, xmm0
IFDEF SWAP_OPERANDS
    ; b + a has the same value, but returns the other payload when both are NaN.
    addss xmm1, xmm2
    movss xmm2, xmm1
ELSE
    addss xmm2, xmm1
ENDIF
    movss xmm4, DWORD PTR [r8]
    mulss xmm2, xmm4
    subss xmm2, xmm3
    movss DWORD PTR [r8+4], xmm2
    divss xmm0, DWORD PTR [r8+4]
    ret
blend_ss ENDP

; double blend_sd(double a, double b, double* c, double d)
; c[1] = (a - b) * c[0] + d, and the result is c[1] / c[0].
blend_sd PROC FRAME
    .endprolog
IFDEF SWAP_OPERANDS
    movsd xmm4, xmm1
    subsd xmm4, xmm0
ELSE
    movsd xmm4, xmm0
    subsd xmm4, xmm1
ENDIF
    mulsd xmm4, QWORD PTR [r8]
    addsd xmm4, xmm3
    movsd QWORD PTR [r8+8], xmm4
    movsd xmm0, QWORD PTR [r8+8]
    divsd xmm0, QWORD PTR [r8]
    ret
blend_sd ENDP

; unsigned convert(float f, double d, void* out)
; out: +0 int32 of f, +4 int32 of d, +8 int64 of f, +16 int64 of d, +24 double of f,
; +32 float of d, +36 the bits of f, +40 the bits of d, +48 int32 of the stored float,
; +56 double of the stored float, +64 the bits of d with the float of d as the low dword,
; +72 the bits of d with f as the low dword.
; The result is the sum of the first two.
convert PROC FRAME
    .endprolog
IFDEF SWAP_OPERANDS
    cvttsd2si eax, xmm1
    cvttss2si ecx, xmm0
ELSE
    cvttss2si eax, xmm0
    cvttsd2si ecx, xmm1
ENDIF
    mov DWORD PTR [r8], eax
    mov DWORD PTR [r8+4], ecx
    add eax, ecx
    cvttss2si rdx, xmm0
    mov QWORD PTR [r8+8], rdx
    cvttsd2si rdx, xmm1
    mov QWORD PTR [r8+16], rdx
    cvtss2sd xmm2, xmm0
    movsd QWORD PTR [r8+24], xmm2
    cvtsd2ss xmm3, xmm1
    movss DWORD PTR [r8+32], xmm3
    movd ecx, xmm0
    movd xmm4, ecx
    movd DWORD PTR [r8+36], xmm4
    movq xmm5, xmm1
    movq QWORD PTR [r8+40], xmm5
    movss xmm5, xmm0
    movq QWORD PTR [r8+72], xmm5
    cvttss2si ecx, DWORD PTR [r8+32]
    mov DWORD PTR [r8+48], ecx
    movq xmm4, QWORD PTR [r8+40]
    cvtsd2ss xmm4, QWORD PTR [r8+40]
    movq QWORD PTR [r8+64], xmm4
    cvtss2sd xmm4, DWORD PTR [r8+32]
    movsd QWORD PTR [r8+56], xmm4
    ret
convert ENDP

; double from_int(int64 n, void* out)
; out: +0 float of the low 32 bits, +4 float of n, +8 double of the low 32 bits,
; +16 double of n, +24 n. The result is the sum of the two doubles.
from_int PROC FRAME
    .endprolog
    xorps xmm0, xmm0
    pxor xmm1, xmm1
IFDEF SWAP_OPERANDS
    cvtsi2ss xmm0, rcx
    cvtsi2ss xmm1, ecx
ELSE
    cvtsi2ss xmm0, ecx
    cvtsi2ss xmm1, rcx
ENDIF
    movss DWORD PTR [rdx], xmm0
    movss DWORD PTR [rdx+4], xmm1
    xorpd xmm2, xmm2
    cvtsi2sd xmm2, ecx
    movsd QWORD PTR [rdx+8], xmm2
    mov QWORD PTR [rdx+24], rcx
    cvtsi2sd xmm0, QWORD PTR [rdx+24]
    movsd QWORD PTR [rdx+16], xmm0
    addsd xmm0, xmm2
    ret
from_int ENDP

; unsigned classify_ss(float a, float b)
; Bit 0: equal. Bit 1: a above b. Bit 2: a below b. Bit 3: unordered.
classify_ss PROC FRAME
    .endprolog
    xor eax, eax
    ucomiss xmm0, xmm1
    jp short ss_unordered
    jne short ss_above
    or eax, 1
ss_above:
    comiss xmm0, xmm1
    jbe short ss_below
    or eax, 2
ss_below:
IFDEF SWAP_OPERANDS
    ucomiss xmm1, xmm0
ELSE
    ucomiss xmm0, xmm1
ENDIF
    jae short ss_done
    or eax, 4
    ret
ss_unordered:
    or eax, 8
ss_done:
    ret
classify_ss ENDP

; unsigned classify_sd(double a, const double* b), with the same bits for a and *b.
classify_sd PROC FRAME
    .endprolog
    xor eax, eax
    ucomisd xmm0, QWORD PTR [rdx]
    jnp short sd_ordered
    or eax, 8
    ret
sd_ordered:
    ucomisd xmm0, QWORD PTR [rdx]
    je short sd_equal
IFDEF SWAP_OPERANDS
    movsd xmm1, QWORD PTR [rdx]
    comisd xmm1, xmm0
ELSE
    comisd xmm0, QWORD PTR [rdx]
ENDIF
    ja short sd_above
    comisd xmm0, QWORD PTR [rdx]
    jb short sd_below
    ret
sd_equal:
    or eax, 1
    ret
sd_above:
    or eax, 2
    ret
sd_below:
    or eax, 4
    ret
classify_sd ENDP

; round_trip(f): CVTSS2SD then CVTSD2SS. Each conversion quiets a signalling NaN, so the pair is
; not an identity. The swapped-operand build keeps the value with MOVSS instead.
round_trip PROC FRAME
    .endprolog
IFDEF SWAP_OPERANDS
    movss xmm0, xmm0
ELSE
    cvtss2sd xmm0, xmm0
    cvtsd2ss xmm0, xmm0
ENDIF
    ret
round_trip ENDP
zero_xorps PROC FRAME
    .endprolog
IFNDEF SWAP_OPERANDS
    xorps xmm0, xmm0
ENDIF
    ret
zero_xorps ENDP

zero_pxor PROC FRAME
    .endprolog
IFNDEF SWAP_OPERANDS
    pxor xmm0, xmm0
ENDIF
    ret
zero_pxor ENDP

zero_xorpd PROC FRAME
    .endprolog
IFNDEF SWAP_OPERANDS
    xorpd xmm0, xmm0
ENDIF
    ret
zero_xorpd ENDP

; unsigned probe_ss(void* out, float a, float b)
; Compare a and b (b and a in the control): equal=0, unordered=1, above=2, below=3.
; Inputs: +0 XMM seed, +8 signed dword in a qword, +16 float.
; Output qwords: +24 after copy, +32 after add, +40 after conversion, +48 after load.
probe_ss PROC FRAME
    .endprolog
    movq xmm3, QWORD PTR [rcx]
    movaps xmm4, xmm3
    movq QWORD PTR [rcx+24], xmm4
    addss xmm4, DWORD PTR [rcx+16]
    movq QWORD PTR [rcx+32], xmm4
    cvtsi2ss xmm4, DWORD PTR [rcx+8]
    movq QWORD PTR [rcx+40], xmm4
    movss xmm4, DWORD PTR [rcx+16]
    movq QWORD PTR [rcx+48], xmm4
    xor eax, eax
IFDEF SWAP_OPERANDS
    comiss xmm2, xmm1
ELSE
    comiss xmm1, xmm2
ENDIF
    jp short probe_unordered
    ja short probe_above
    jb short probe_below
    ret
probe_unordered:
    or eax, 1
    ret
probe_above:
    or eax, 2
    ret
probe_below:
    or eax, 3
    ret
probe_ss ENDP

; unsigned neg_jcc(int32_t x)
; Bit 0: -x != 0 (NZ after NEG). Bit 1: carry after NEG (inverted in control).
; Bit 2: -x < 0 (SF after NEG). Bit 3: x == INT32_MIN (OF after NEG).
neg_jcc PROC FRAME
    .endprolog
    xor eax, eax
    mov edx, ecx
    neg edx
    je short nj_zero
    or eax, 1
nj_zero:
    mov edx, ecx
    neg edx
IFDEF SWAP_OPERANDS
    jae short nj_carry
ELSE
    jb short nj_carry
ENDIF
    jmp short nj_sign
nj_carry:
    or eax, 2
nj_sign:
    mov edx, ecx
    neg edx
    js short nj_neg
    jmp short nj_ovf
nj_neg:
    or eax, 4
nj_ovf:
    mov edx, ecx
    neg edx
    jo short nj_overflow
    ret
nj_overflow:
    or eax, 8
    ret
neg_jcc ENDP

; uint64_t setcc_mix(uint64_t a, uint64_t b, uint8_t* out)
; Seeds RAX (non-zero AL) so SETcc overwrites AL on both true and false outcomes while retaining bits 8..63, and records CMP/NEG flags to memory.
setcc_mix PROC FRAME
    .endprolog
    mov rax, 123456789abcde5ah
    cmp ecx, edx
    mov r9d, eax
IFDEF SWAP_OPERANDS
    setb al
ELSE
    setl al
ENDIF
    sete BYTE PTR [r8]
    neg ecx
    setb BYTE PTR [r8+1]
    seto BYTE PTR [r8+2]
    sete BYTE PTR [r8+3]
    ret
setcc_mix ENDP

; uint64_t cmovcc_mix(uint64_t a, uint64_t b, const uint64_t* mem)
; Covers 32-bit zero-extending CMOVcc and 64-bit CMOVcc after CMP and NEG.
cmovcc_mix PROC FRAME
    .endprolog
    mov rax, 0fedcba9876543210h
    cmp ecx, edx
IFDEF SWAP_OPERANDS
    cmovle eax, DWORD PTR [r8]
ELSE
    cmovg eax, DWORD PTR [r8]
ENDIF
    neg rcx
    cmovo rax, QWORD PTR [r8+8]
    cmove rax, rdx
    ret
cmovcc_mix ENDP

; float min_ss(float a, float b, uint8_t* c)
; c[0..8) seeds the destination registers and is not written. c+8 gets the register form on a and b, c+17
; and c+21 get a and b, c+28 gets the memory form (an unaligned dword source). The result is the register form.
min_ss PROC FRAME
    .endprolog
    movss DWORD PTR [r8+17], xmm0
    movss DWORD PTR [r8+21], xmm1
    movq xmm2, QWORD PTR [r8]
    movq xmm3, QWORD PTR [r8]
IFDEF SWAP_OPERANDS
    movss xmm2, xmm1
    minss xmm2, xmm0
    movss xmm3, xmm1
    minss xmm3, DWORD PTR [r8+17]
ELSE
    movss xmm2, xmm0
    minss xmm2, xmm1
    movss xmm3, xmm0
    minss xmm3, DWORD PTR [r8+21]
ENDIF
    movq QWORD PTR [r8+8], xmm2
    movq QWORD PTR [r8+28], xmm3
    movaps xmm0, xmm2
    ret
min_ss ENDP

; float max_ss(float a, float b, uint8_t* c)
; c[0..8) seeds the destination registers and is not written. c+8 gets the register form on a and b, c+17
; and c+21 get a and b, c+28 gets the memory form (an unaligned dword source). The result is the register form.
max_ss PROC FRAME
    .endprolog
    movss DWORD PTR [r8+17], xmm0
    movss DWORD PTR [r8+21], xmm1
    movq xmm2, QWORD PTR [r8]
    movq xmm3, QWORD PTR [r8]
IFDEF SWAP_OPERANDS
    movss xmm2, xmm1
    maxss xmm2, xmm0
    movss xmm3, xmm1
    maxss xmm3, DWORD PTR [r8+17]
ELSE
    movss xmm2, xmm0
    maxss xmm2, xmm1
    movss xmm3, xmm0
    maxss xmm3, DWORD PTR [r8+21]
ENDIF
    movq QWORD PTR [r8+8], xmm2
    movq QWORD PTR [r8+28], xmm3
    movaps xmm0, xmm2
    ret
max_ss ENDP

; double min_sd(double a, double b, uint8_t* c)
; c+8 gets the register form on a and b, c+17 and c+25 get a and b, c+36 gets the memory form (an unaligned
; qword source). The result is the register form.
min_sd PROC FRAME
    .endprolog
    movsd QWORD PTR [r8+17], xmm0
    movsd QWORD PTR [r8+25], xmm1
IFDEF SWAP_OPERANDS
    movsd xmm2, xmm1
    minsd xmm2, xmm0
    movsd xmm3, xmm1
    minsd xmm3, QWORD PTR [r8+17]
ELSE
    movsd xmm2, xmm0
    minsd xmm2, xmm1
    movsd xmm3, xmm0
    minsd xmm3, QWORD PTR [r8+25]
ENDIF
    movsd QWORD PTR [r8+8], xmm2
    movsd QWORD PTR [r8+36], xmm3
    movsd xmm0, xmm2
    ret
min_sd ENDP

; double max_sd(double a, double b, uint8_t* c)
; c+8 gets the register form on a and b, c+17 and c+25 get a and b, c+36 gets the memory form (an unaligned
; qword source). The result is the register form.
max_sd PROC FRAME
    .endprolog
    movsd QWORD PTR [r8+17], xmm0
    movsd QWORD PTR [r8+25], xmm1
IFDEF SWAP_OPERANDS
    movsd xmm2, xmm1
    maxsd xmm2, xmm0
    movsd xmm3, xmm1
    maxsd xmm3, QWORD PTR [r8+17]
ELSE
    movsd xmm2, xmm0
    maxsd xmm2, xmm1
    movsd xmm3, xmm0
    maxsd xmm3, QWORD PTR [r8+25]
ENDIF
    movsd QWORD PTR [r8+8], xmm2
    movsd QWORD PTR [r8+36], xmm3
    movsd xmm0, xmm2
    ret
max_sd ENDP

; float root_ss(float a, uint8_t* c)
; c[0..8) seeds the destination registers and is not written. c+8 gets the register form, c+17 gets a and c+24
; gets the memory form (an unaligned dword source). The result is the register form. The control zeroes the
; seed first, so the bits above the result differ.
root_ss PROC FRAME
    .endprolog
    movss DWORD PTR [rdx+17], xmm0
    movq xmm2, QWORD PTR [rdx]
    movq xmm3, QWORD PTR [rdx]
IFDEF SWAP_OPERANDS
    xorps xmm2, xmm2
ENDIF
    sqrtss xmm2, xmm0
    sqrtss xmm3, DWORD PTR [rdx+17]
    movq QWORD PTR [rdx+8], xmm2
    movq QWORD PTR [rdx+24], xmm3
    movaps xmm0, xmm2
    ret
root_ss ENDP

; double root_sd(double a, uint8_t* c)
; c+8 gets the register form, c+17 gets a and c+32 gets the memory form (an unaligned qword source). The result
; is the register form. The control copies a instead of taking its root.
root_sd PROC FRAME
    .endprolog
    movsd QWORD PTR [rdx+17], xmm0
IFDEF SWAP_OPERANDS
    movsd xmm2, xmm0
ELSE
    sqrtsd xmm2, xmm0
ENDIF
    sqrtsd xmm3, QWORD PTR [rdx+17]
    movsd QWORD PTR [rdx+8], xmm2
    movsd QWORD PTR [rdx+32], xmm3
    movsd xmm0, xmm2
    ret
root_sd ENDP

; float root_const_ss(void) is the square root of 2 and double root_const_sd(void) is the square root of 3.
; The operand is built from an integer constant, so a translation that lets the compiler evaluate the root
; would use the default rounding and not the rounding of the host MXCSR. The control copies the operand.
root_const_ss PROC FRAME
    .endprolog
    mov eax, 2
    xorps xmm0, xmm0
    cvtsi2ss xmm0, eax
IFNDEF SWAP_OPERANDS
    sqrtss xmm0, xmm0
ENDIF
    ret
root_const_ss ENDP

root_const_sd PROC FRAME
    .endprolog
    mov eax, 3
    xorpd xmm0, xmm0
    cvtsi2sd xmm0, eax
IFNDEF SWAP_OPERANDS
    sqrtsd xmm0, xmm0
ENDIF
    ret
root_const_sd ENDP

; uint64_t round_cvt(float f, double d, uint8_t* out)
; CVTSS2SI and CVTSD2SI into 32-bit and 64-bit registers from a register and from memory; the 32-bit forms start from
; all ones, so a destination that keeps its upper half shows. The words of out hold the results; f and d sit at +40
; and +48. SWAP_OPERANDS converts d where the first conversion converts f.
round_cvt PROC FRAME
    .endprolog
    mov rax, -1
    mov rcx, -1
    mov r9, -1
    mov r10, -1
    movss DWORD PTR [r8+40], xmm0
    movsd QWORD PTR [r8+48], xmm1
IFDEF SWAP_OPERANDS
    cvtsd2si eax, xmm1
ELSE
    cvtss2si eax, xmm0
ENDIF
    cvtsd2si ecx, xmm1
    cvtss2si rdx, xmm0
    cvtsd2si r11, xmm1
    cvtss2si r9d, DWORD PTR [r8+40]
    cvtsd2si r10d, QWORD PTR [r8+48]
    mov QWORD PTR [r8], rax
    mov QWORD PTR [r8+8], rcx
    mov QWORD PTR [r8+16], rdx
    mov QWORD PTR [r8+24], r11
    mov QWORD PTR [r8+32], r9
    mov QWORD PTR [r8+56], r10
    cvtss2si rdx, DWORD PTR [r8+40]
    mov QWORD PTR [r8+64], rdx
    cvtsd2si rdx, QWORD PTR [r8+48]
    mov QWORD PTR [r8+72], rdx
    ret
round_cvt ENDP

; uint64_t movq_gpr(double d, uint64_t q)
; Round-trips 64-bit values across XMM and 64-bit GPRs via MOVQ (encoded with DB bytes
; for MASM portability across ML64 versions). Returns d_bits XOR q (or d_bits + q in
; the SWAP_OPERANDS control), preserving all 64 bits across -0.0, NaNs, subnormals, and +/-inf.
movq_gpr PROC FRAME
    .endprolog
    ; movq r9, xmm0
    DB 66h, 49h, 0Fh, 7Eh, 0C1h
    ; movq xmm2, r9
    DB 66h, 49h, 0Fh, 6Eh, 0D1h
    ; movq rax, xmm2
    DB 66h, 48h, 0Fh, 7Eh, 0D0h
    ; movq xmm3, rdx
    DB 66h, 48h, 0Fh, 6Eh, 0DAh
    ; movq r10, xmm3
    DB 66h, 49h, 0Fh, 7Eh, 0DAh
IFDEF SWAP_OPERANDS
    add rax, r10
ELSE
    xor rax, r10
ENDIF
    ret
movq_gpr ENDP

END
