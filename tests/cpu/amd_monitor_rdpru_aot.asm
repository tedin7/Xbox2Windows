; coverage: record cpu_aot_amd_monitor_rdpru
; Source-owned differential fixture for AMD Zen 2 user-mode MONITORX, MWAITX, and RDPRU.
;   RCX = uint64_t input_a (preset value or caller-owned buffer pointer)
;   RDX = uint64_t input_b (preset value or caller-owned buffer pointer)
;   R8  = void*    output  (64-byte caller-owned output buffer)
;   RAX = uint64_t return value
; Note: ML64 emits MONITORX (0F 01 FA), MWAITX (0F 01 FB), and RDPRU (0F 01 FD) via
; exact opcode bytes where older MASM lacks mnemonics, decoded natively by iced-x86.
.code

; RDPRU with ECX=0 (MPERF): zero-extends EAX->RAX & EDX->RDX, preserves RCX and RFLAGS.
; Output [r8]: +0=RAX, +8=RDX, +16=RCX, +24=CF|(ZF<<1)|(SF<<2)|(OF<<3). Returns (RDX<<32)|RAX.
op_rdpru_mperf PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    mov rax, -1
    mov rdx, -1
IFDEF ALTER_AMD_EXT
    mov rcx, 1
ELSE
    mov rcx, 0
ENDIF
    add r9, r10
    db 0fh, 01h, 0fdh       ; rdpru
    mov qword ptr [r8], rax
    mov qword ptr [r8+8], rdx
    mov qword ptr [r8+16], rcx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    shl rdx, 32
    or rax, rdx
    ret
op_rdpru_mperf ENDP

; RDPRU with ECX=1 (APERF): same ABI and flag/register preservation checks.
op_rdpru_aperf PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    mov rax, -1
    mov rdx, -1
IFDEF ALTER_AMD_EXT
    mov rcx, 0
ELSE
    mov rcx, 1
ENDIF
    cmp r9, r10
    db 0fh, 01h, 0fdh       ; rdpru
    mov qword ptr [r8], rax
    mov qword ptr [r8+8], rdx
    mov qword ptr [r8+16], rcx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    shl rdx, 32
    or rax, rdx
    ret
op_rdpru_aperf ENDP

; Two consecutive RDPRU reads: verifies monotonicity and prevents /O2 CSE merging.
; Output [r8]: +0=first_rax, +8=first_rdx, +16=second_rax, +24=second_rdx, +32=rcx.
op_rdpru_twice PROC FRAME
    .endprolog
    xor ecx, ecx
    db 0fh, 01h, 0fdh       ; rdpru (1)
    mov qword ptr [r8], rax
    mov qword ptr [r8+8], rdx
IFDEF ALTER_AMD_EXT
    mov ecx, 1
ENDIF
    db 0fh, 01h, 0fdh       ; rdpru (2)
    mov qword ptr [r8+16], rax
    mov qword ptr [r8+24], rdx
    mov qword ptr [r8+32], rcx
    shl rdx, 32
    or rax, rdx
    ret
op_rdpru_twice ENDP

; Overwritten RDPRU read: prevents /O2 DCE of the counter read while returning constant 7.
op_rdpru_unused PROC FRAME
    .endprolog
    xor ecx, ecx
    db 0fh, 01h, 0fdh       ; rdpru
IFDEF ALTER_AMD_EXT
    mov eax, 42
ELSE
    mov eax, 7
ENDIF
    xor edx, edx
    ret
op_rdpru_unused ENDP

; MONITORX on [r8] + triggering store + MWAITX (ECX=0, EAX=0): preserves GPRs, RFLAGS, and stores.
; Output [r8]: +0=rcx, +8=rdx, +16=rax_hint, +24=flags_mask, +32=ecx_ext.
op_monitorx_mwaitx_r8 PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    lea rax, [r8]
    xor ecx, ecx
    xor edx, edx
    db 0fh, 01h, 0fah       ; monitorx (arms [r8], ECX=0, EDX=0)
    mov qword ptr [r8], r9  ; observable store (also wakes monitored line on AMD hardware)
    mov eax, 0
    mov ecx, 0
    cmp r9, r10             ; preset CF/ZF/SF/OF before MWAITX
    db 0fh, 01h, 0fbh       ; mwaitx (hints=0, ext=0; preserves GPRs & RFLAGS)
IFDEF ALTER_AMD_EXT
    mov qword ptr [r8+8], r9
ELSE
    mov qword ptr [r8+8], r10
ENDIF
    mov qword ptr [r8+16], rax
    mov qword ptr [r8+32], rcx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    mov rax, rcx
    ret
op_monitorx_mwaitx_r8 ENDP

; MONITORX + MWAITX with C1 hint (EAX=0x10) and interrupt-break extension (ECX=1, no timer bit).
op_monitorx_mwaitx_intbrk PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    lea rax, [r8]
    mov ecx, 0
    mov edx, 0
    db 0fh, 01h, 0fah       ; monitorx
    mov qword ptr [r8], r9
IFDEF ALTER_AMD_EXT
    mov eax, 20h
ELSE
    mov eax, 10h
ENDIF
    mov ecx, 1              ; ECX[0]=1 interrupt-break (ECX[1]=0 no timer/EBX)
    add r9, r10
    db 0fh, 01h, 0fbh       ; mwaitx
    mov qword ptr [r8+8], rax
    mov qword ptr [r8+16], rcx
    mov qword ptr [r8+24], rdx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+32], rcx
    mov rax, rcx
    ret
op_monitorx_mwaitx_intbrk ENDP

; MONITORX on flat [rcx] and [rdx] pointers; verifies pre/post store ordering and NULL fault ordering.
op_monitorx_rcx_rdx PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    mov qword ptr [r8], r10 ; pre-MONITORX store
    lea rax, [rcx]
    mov ecx, 0
    mov edx, 0
    cmp r9, r10
    db 0fh, 01h, 0fah       ; monitorx on [rcx] (faults if rcx == NULL)
    mov qword ptr [r8+8], rax
    mov rax, r10            ; flat pointer copy of original [rdx]
    db 0fh, 01h, 0fah       ; monitorx on [rdx] (faults if rdx == NULL)
    mov qword ptr [r8+16], rax
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    mov rax, rcx
    ret
op_monitorx_rcx_rdx ENDP

; 32-bit ADD flag path preserved across RDPRU (ECX=0); upper 32 bits of inputs ignored.
op_add32_rdpru PROC FRAME
    .endprolog
    mov r9d, ecx
    mov r10d, edx
    mov rax, -1
    mov rdx, -1
    mov ecx, 0
    add r9d, r10d
    db 0fh, 01h, 0fdh       ; rdpru
    mov qword ptr [r8], rax
    mov qword ptr [r8+8], rdx
    mov qword ptr [r8+16], rcx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    shl rdx, 32
    or rax, rdx
    ret
op_add32_rdpru ENDP

; 32-bit CMP flag path preserved across MONITORX/MWAITX; upper 32 bits of inputs ignored.
op_cmp32_mwaitx PROC FRAME
    .endprolog
    mov r9d, ecx
    mov r10d, edx
    lea rax, [r8]
    xor ecx, ecx
    xor edx, edx
    db 0fh, 01h, 0fah       ; monitorx
    mov qword ptr [r8], r9
    mov eax, 0
    mov ecx, 0
    cmp r9d, r10d
    db 0fh, 01h, 0fbh       ; mwaitx
    mov qword ptr [r8+8], rax
    mov qword ptr [r8+16], rcx
    mov ecx, 0
    mov r9d, 0
    mov r10d, 0
    mov r11d, 0
    setb cl
    sete r9b
    sets r10b
    seto r11b
    shl r9d, 1
    or ecx, r9d
    shl r10d, 2
    or ecx, r10d
    shl r11d, 3
    or ecx, r11d
    mov qword ptr [r8+24], rcx
    mov rax, rcx
    ret
op_cmp32_mwaitx ENDP

END
