; coverage: record cpu_aot_seg_inspect
; Source-built MASM fixture for LAR, LSL, VERR, and VERW segment inspection
; leaf functions (Windows x64 ABI: RCX, RDX, R8 -> RAX), plus a native-only
; segment register probe helper.
.code

; --- Native-only helper (never translated) ---
; Writes current cs, ss, ds, es, gs into five WORD slots at [rcx].
native_read_segs PROC FRAME
    .endprolog
    mov ax, cs
    mov WORD PTR [rcx], ax
    mov ax, ss
    mov WORD PTR [rcx+2], ax
    mov ax, ds
    mov WORD PTR [rcx+4], ax
    mov ax, es
    mov WORD PTR [rcx+6], ax
    mov ax, gs
    mov WORD PTR [rcx+8], ax
    xor eax, eax
    ret
native_read_segs ENDP

; --- Translated exports ---
; Register LAR forms: rcx = selector, rdx = 64-bit sentinel, r8 = out buffer.
lar_reg16 PROC FRAME
    .endprolog
    mov rax, rdx
    lar ax, cx
    sete BYTE PTR [r8]
    ret
lar_reg16 ENDP

lar_reg32 PROC FRAME
    .endprolog
    mov rax, rdx
    lar eax, ecx
    sete BYTE PTR [r8]
    ret
lar_reg32 ENDP

lar_reg64 PROC FRAME
    .endprolog
    mov rax, rdx
    lar rax, rcx
    sete BYTE PTR [r8]
    ret
lar_reg64 ENDP

; Memory LAR forms: rcx = pointer to selector buffer, rdx = sentinel, r8 = out buffer.
lar_mem16 PROC FRAME
    .endprolog
    mov rax, rdx
    lar ax, WORD PTR [rcx]
    sete BYTE PTR [r8]
    ret
lar_mem16 ENDP

lar_mem32_disp PROC FRAME
    .endprolog
    mov rax, rdx
    lar eax, WORD PTR [rcx+4]
    sete BYTE PTR [r8]
    ret
lar_mem32_disp ENDP

lar_mem64_disp PROC FRAME
    .endprolog
    mov rax, rdx
    lar rax, WORD PTR [rcx+8]
    sete BYTE PTR [r8]
    ret
lar_mem64_disp ENDP

; Register LSL forms: rcx = selector, rdx = 64-bit sentinel, r8 = out buffer.
lsl_reg16 PROC FRAME
    .endprolog
    mov rax, rdx
    lsl ax, cx
    sete BYTE PTR [r8]
    ret
lsl_reg16 ENDP

lsl_reg32 PROC FRAME
    .endprolog
    mov rax, rdx
    lsl eax, ecx
    sete BYTE PTR [r8]
    ret
lsl_reg32 ENDP

lsl_reg64 PROC FRAME
    .endprolog
    mov rax, rdx
    lsl rax, rcx
    sete BYTE PTR [r8]
    ret
lsl_reg64 ENDP

; Memory LSL forms: rcx = pointer to selector buffer, rdx = sentinel, r8 = out buffer.
lsl_mem16 PROC FRAME
    .endprolog
    mov rax, rdx
    lsl ax, WORD PTR [rcx]
    sete BYTE PTR [r8]
    ret
lsl_mem16 ENDP

lsl_mem32_disp PROC FRAME
    .endprolog
    mov rax, rdx
    lsl eax, WORD PTR [rcx+4]
    sete BYTE PTR [r8]
    ret
lsl_mem32_disp ENDP

lsl_mem64_disp PROC FRAME
    .endprolog
    mov rax, rdx
    lsl rax, WORD PTR [rcx+8]
    sete BYTE PTR [r8]
    ret
lsl_mem64_disp ENDP

; VERR / VERW forms: verify ZF without touching other GPRs.
; ALTER_IMM swaps VERR <-> VERW so mutant divergence is directly observed.
verr_reg PROC FRAME
    .endprolog
    mov rax, rdx
    mov r9, 0CAFEBABE12345678h
IFDEF ALTER_IMM
    verw cx
ELSE
    verr cx
ENDIF
    sete BYTE PTR [r8]
    mov QWORD PTR [r8+8], r9
    ret
verr_reg ENDP

verw_reg PROC FRAME
    .endprolog
    mov rax, rdx
    mov r9, 0CAFEBABE12345678h
IFDEF ALTER_IMM
    verr cx
ELSE
    verw cx
ENDIF
    sete BYTE PTR [r8]
    mov QWORD PTR [r8+8], r9
    ret
verw_reg ENDP

verr_mem_disp PROC FRAME
    .endprolog
    mov rax, rdx
    mov r9, 0DEADBEEF00112233h
IFDEF ALTER_IMM
    verw WORD PTR [rcx+2]
ELSE
    verr WORD PTR [rcx+2]
ENDIF
    sete BYTE PTR [r8]
    mov QWORD PTR [r8+8], r9
    ret
verr_mem_disp ENDP

verw_mem_disp PROC FRAME
    .endprolog
    mov rax, rdx
    mov r9, 0DEADBEEF00112233h
IFDEF ALTER_IMM
    verr WORD PTR [rcx+2]
ELSE
    verw WORD PTR [rcx+2]
ENDIF
    sete BYTE PTR [r8]
    mov QWORD PTR [r8+8], r9
    ret
verw_mem_disp ENDP

; Immediate selector mutant: ALTER_IMM changes 02Bh (writable user data) to 033h (64-bit user code).
seg_imm_select PROC FRAME
    .endprolog
    mov rax, rdx
    mov r10, rdx
IFDEF ALTER_IMM
    mov r9d, 033h
ELSE
    mov r9d, 02Bh
ENDIF
    lar rax, r9
    lsl r10, r9
    verr r9w
    sete BYTE PTR [r8]
    verw r9w
    sete BYTE PTR [r8+1]
    mov QWORD PTR [r8+8], r10
    ret
seg_imm_select ENDP

; Preserves CF, SF, OF, PF while updating ZF; ALTER_IMM inverts ZF and corrupts untouched dst.
seg_flags_preserve PROC FRAME
    .endprolog
    mov r9d, 080000001h
    add r9d, 080000002h     ; r9d = 3 -> CF=1, OF=1, SF=0, ZF=0, PF=1
    mov rax, rdx
    lar eax, ecx
IFDEF ALTER_IMM
    setne BYTE PTR [r8]
    cmovne rax, r9
ELSE
    sete BYTE PTR [r8]
ENDIF
    setb BYTE PTR [r8+1]
    sets BYTE PTR [r8+2]
    seto BYTE PTR [r8+3]
    setp BYTE PTR [r8+4]
    ret
seg_flags_preserve ENDP

; Tests 32-bit CMOVE after LAR: even on ZF=0, cmove eax, r9d zeroes RAX[63:32].
seg_cmov32 PROC FRAME
    .endprolog
    mov rax, rdx
    mov r9d, 012345678h
    mov r10, rdx
    lar r10d, ecx
    cmove eax, r9d
    sete BYTE PTR [r8]
    ret
seg_cmov32 ENDP

; Discarded m16 reads: must still fault on invalid [rcx] even under /O2.
seg_discarded_read PROC FRAME
    .endprolog
    mov r9, rdx
    mov r10, rdx
    lar r9d, WORD PTR [rcx]
    lsl r10d, WORD PTR [rcx]
    verr WORD PTR [rcx]
    verw WORD PTR [rcx]
    mov rax, rdx
    ret
seg_discarded_read ENDP

END
