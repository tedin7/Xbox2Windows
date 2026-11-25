; coverage: source cpu_aot_x87_examine
; Source-owned differential fixture for FTST, FXAM, FICOM, FICOMP, FNOP, FNCLEX and FCLEX, and the
; 8087/287 controls FENI, FNENI, FDISI, FNDISI and FSETPM (ML64 has no FNSETPM spelling and assembles
; FSETPM without a WAIT), FFREEP ST(0), FFREE of an empty register and FSTSW AX, in the x87 leaf-function profile. Each export reads an input block at RCX (an m80 value at +0, an m16 integer at
; +16 and an m32 integer at +24), records C0, C2 and C3 after each status-word writer as bytes of the
; output block at RDX, and returns with the x87 stack empty.
.code

ftst_fxam PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    ftst
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+0], 1
    jne tst_c0
    mov byte ptr [rdx+0], 0
tst_c0:
    test ah, 004h
    mov byte ptr [rdx+1], 1
    jne tst_c2
    mov byte ptr [rdx+1], 0
tst_c2:
    test ah, 040h
    mov byte ptr [rdx+2], 1
    jne tst_c3
    mov byte ptr [rdx+2], 0
tst_c3:
    fxam
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+3], 1
    jne xam_c0
    mov byte ptr [rdx+3], 0
xam_c0:
    test ah, 004h
    mov byte ptr [rdx+4], 1
    jne xam_c2
    mov byte ptr [rdx+4], 0
xam_c2:
    test ah, 040h
    mov byte ptr [rdx+5], 1
    jne xam_c3
    mov byte ptr [rdx+5], 0
xam_c3:
    fstp tbyte ptr [rdx+16]
    fxam
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+6], 1
    jne empty_c0
    mov byte ptr [rdx+6], 0
empty_c0:
    test ah, 004h
    mov byte ptr [rdx+7], 1
    jne empty_c2
    mov byte ptr [rdx+7], 0
empty_c2:
    test ah, 040h
    mov byte ptr [rdx+8], 1
    jne empty_c3
    mov byte ptr [rdx+8], 0
empty_c3:
    mov eax, 0
    ret
ftst_fxam ENDP

ficom_m16 PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    ficom word ptr [rcx+16]
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+0], 1
    jne icom16_c0
    mov byte ptr [rdx+0], 0
icom16_c0:
    test ah, 004h
    mov byte ptr [rdx+1], 1
    jne icom16_c2
    mov byte ptr [rdx+1], 0
icom16_c2:
    test ah, 040h
    mov byte ptr [rdx+2], 1
    jne icom16_c3
    mov byte ptr [rdx+2], 0
icom16_c3:
    ficomp word ptr [rcx+16]
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+3], 1
    jne icomp16_c0
    mov byte ptr [rdx+3], 0
icomp16_c0:
    test ah, 004h
    mov byte ptr [rdx+4], 1
    jne icomp16_c2
    mov byte ptr [rdx+4], 0
icomp16_c2:
    test ah, 040h
    mov byte ptr [rdx+5], 1
    jne icomp16_c3
    mov byte ptr [rdx+5], 0
icomp16_c3:
    mov eax, 0
    ret
ficom_m16 ENDP

ficom_m32 PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    ficom dword ptr [rcx+24]
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+0], 1
    jne icom32_c0
    mov byte ptr [rdx+0], 0
icom32_c0:
    test ah, 004h
    mov byte ptr [rdx+1], 1
    jne icom32_c2
    mov byte ptr [rdx+1], 0
icom32_c2:
    test ah, 040h
    mov byte ptr [rdx+2], 1
    jne icom32_c3
    mov byte ptr [rdx+2], 0
icom32_c3:
    ficomp dword ptr [rcx+24]
    fnstsw ax
    test ah, 001h
    mov byte ptr [rdx+3], 1
    jne icomp32_c0
    mov byte ptr [rdx+3], 0
icomp32_c0:
    test ah, 004h
    mov byte ptr [rdx+4], 1
    jne icomp32_c2
    mov byte ptr [rdx+4], 0
icomp32_c2:
    test ah, 040h
    mov byte ptr [rdx+5], 1
    jne icomp32_c3
    mov byte ptr [rdx+5], 0
icomp32_c3:
    mov eax, 0
    ret
ficom_m32 ENDP

fnop_fclex PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    fnop
    fnclex
    fclex
    fstp tbyte ptr [rdx+16]
    mov eax, 0
    ret
fnop_fclex ENDP

; The 287 and later execute these as FNOP: the stored value, the control word and the empty stack on
; return must match a plain load and store.
legacy_controls PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    feni
    fneni
    fdisi
    fndisi
    fsetpm
    fstp tbyte ptr [rdx+16]
    mov eax, 0
    ret
legacy_controls ENDP

; FFREEP ST(0) pops without storing: physical TOP after each FFREEP (FNSTSW AX, AH masked to TOP) and the
; values left below it must match. Output: TOP bytes (TOP << 3) at +0 (before), +1, +2 and +3 (after each
; FFREEP, the last at depth 1 down to an empty stack), 1.0 at +16 and the input m80 at +32.
ffreep_top PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    fld1
    fldz
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+0], ah
    ffreep st(0)
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+1], ah
    fstp tbyte ptr [rdx+16]
    fld1
    ffreep st(0)
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+2], ah
    fstp tbyte ptr [rdx+32]
    fldz
    ffreep st(0)
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+3], ah
    mov eax, 0
    ret
ffreep_top ENDP

; FFREE of an already empty register changes neither occupancy nor TOP. Output: TOP bytes (TOP << 3) at +0
; (depth 0, after ffree st(0) and st(7)), +1 (depth 1, after ffree st(1), st(4) and st(7)) and +2 (depth 8,
; after ffree st(7) at depth 7 and one more load, which must not overflow), then 0.0 at +16, 1.0 at +26 and the
; input m80 at +40.
ffree_empty PROC FRAME
    .endprolog
    ffree st(0)
    ffree st(7)
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+0], ah
    fld tbyte ptr [rcx]
    ffree st(1)
    ffree st(4)
    ffree st(7)
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+1], ah
    fld1
    fld1
    fld1
    fld1
    fld1
    fld1
    ffree st(7)
    fldz
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+2], ah
    fstp tbyte ptr [rdx+16]
    fstp tbyte ptr [rdx+26]
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp tbyte ptr [rdx+40]
    mov eax, 0
    ret
ffree_empty ENDP

; FSTSW AX assembles as WAIT and FNSTSW AX. WAIT leaves C0 to C3 undefined, so the C0, C2 and C3 that FTST set
; before the last read are unknown and only TOP is: AH is masked to TOP. Output: TOP bytes (TOP << 3) at +0
; (depth 0), +1 (depth 1) and +2 (depth 2, after FTST), and the input m80 at +16.
fstsw_top PROC FRAME
    .endprolog
    fstsw ax
    and ah, 38h
    mov byte ptr [rdx+0], ah
    fld tbyte ptr [rcx]
    fstsw ax
    and ah, 38h
    mov byte ptr [rdx+1], ah
    fld1
    ftst
    fstsw ax
    and ah, 38h
    mov byte ptr [rdx+2], ah
    fstp st(0)
    fstp tbyte ptr [rdx+16]
    mov eax, 0
    ret
fstsw_top ENDP

; FFREE ST(0) directly followed by FINCSTP pops ST(0) without storing it. Output: TOP bytes (TOP << 3) at +0 (after
; the pair at depth 1), +1 (after the pair at depth 2, which drops 1.0 above the input), +2 (after the pair at
; depth 8, which drops pi) and +3 (after one more load, which must not overflow), then the pushed 1.0 at +16, the
; 0.0 below it at +26, the 1.0 below the zeros at +36 and the input m80 at +46.
ffree_fincstp PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    ffree st(0)
    fincstp
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+0], ah
    fld tbyte ptr [rcx]
    fld1
    ffree st(0)
    fincstp
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+1], ah
    fld1
    fldz
    fldz
    fldz
    fldz
    fldz
    fldpi
    ffree st(0)
    fincstp
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+2], ah
    fld1
    fnstsw ax
    and ah, 38h
    mov byte ptr [rdx+3], ah
    fstp tbyte ptr [rdx+16]
    fstp tbyte ptr [rdx+26]
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp tbyte ptr [rdx+36]
    fstp tbyte ptr [rdx+46]
    mov eax, 0
    ret
ffree_fincstp ENDP

END
