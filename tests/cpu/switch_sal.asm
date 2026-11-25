; coverage: record cpu_aot_switch_sal
; Source-built differential fixture and negative controls for SAL (/6 encoding alias of SHL) in
; tools/cpu_switch_aot.py. ML64 assembles the SAL mnemonic as /4 (SHL), so SAL instructions are
; emitted as explicit /6 byte sequences and recorded via iced_x86.Mnemonic.
; Every differential export takes RCX = x, RDX = y and R8 = address of three writable QWORDs.
.code

; --- Register SAL by CL and 1 (all four widths) ---

sal_reg_cl_8 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0D2h, 0F0h                   ; sal al, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    db 041h, 0D2h, 0F1h             ; sal r9b, cl
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    db 041h, 0D2h, 0F2h             ; sal r10b, cl
    db 041h, 0D0h, 0F2h             ; sal r10b, 1
    mov QWORD PTR [r8+16], r10
    ret
sal_reg_cl_8 ENDP

sal_reg_cl_16 PROC FRAME
    .endprolog
    mov rax, rdx
    db 066h, 0D3h, 0F0h             ; sal ax, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    db 066h, 041h, 0D3h, 0F1h       ; sal r9w, cl
    mov QWORD PTR [r8+8], r9
    mov QWORD PTR [r8+16], rdx
    ret
sal_reg_cl_16 ENDP

sal_reg_cl_32 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0D3h, 0F0h                   ; sal eax, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    db 041h, 0D3h, 0F1h             ; sal r9d, cl
    mov QWORD PTR [r8+8], r9
    mov QWORD PTR [r8+16], rdx
    ret
sal_reg_cl_32 ENDP

sal_reg_cl_64 PROC FRAME
    .endprolog
    mov rax, rdx
    db 048h, 0D3h, 0F0h             ; sal rax, cl
    mov QWORD PTR [r8], rax
    mov r10, rdx
    db 049h, 0D3h, 0F2h             ; sal r10, cl
    mov QWORD PTR [r8+8], r10
    mov QWORD PTR [r8+16], rdx
    ret
sal_reg_cl_64 ENDP

; --- Memory SAL by CL and immediate (non-flag-producing, including count=0 and count>=width) ---

sal_mem_8_16 PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    db 041h, 0D2h, 030h             ; sal BYTE PTR [r8], cl
    db 041h, 0C0h, 070h, 008h, 003h ; sal BYTE PTR [r8+8], 3
    db 066h, 041h, 0D3h, 070h, 010h ; sal WORD PTR [r8+16], cl
    mov rax, rdx
    db 066h, 0D1h, 0F0h             ; sal ax, 1
    ret
sal_mem_8_16 ENDP

sal_mem_32_64 PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    db 041h, 0D3h, 030h             ; sal DWORD PTR [r8], cl
    db 049h, 0D3h, 070h, 008h       ; sal QWORD PTR [r8+8], cl
    db 049h, 0D1h, 070h, 010h       ; sal QWORD PTR [r8+16], 1
    mov rax, QWORD PTR [r8+16]
    ret
sal_mem_32_64 ENDP

sal_mem_imm_edge PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    db 041h, 0C0h, 030h, 000h       ; sal BYTE PTR [r8], 0
    db 041h, 0C0h, 070h, 008h, 008h ; sal BYTE PTR [r8+8], 8
    db 066h, 041h, 0C1h, 070h, 010h, 010h ; sal WORD PTR [r8+16], 16
    mov rax, rcx
    db 0C1h, 0F0h, 020h             ; sal eax, 32
    ret
sal_mem_imm_edge ENDP

; --- Flag-producing SAL with count=1 (all 16 SETcc conditions across all four widths) ---

sal1_8 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0D0h, 0F0h                   ; sal al, 1
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov QWORD PTR [r8+16], rax
    ret
sal1_8 ENDP

sal1_16 PROC FRAME
    .endprolog
    mov r9, rdx
    db 066h, 041h, 0D1h, 0F1h       ; sal r9w, 1
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov QWORD PTR [r8+16], r9
    mov rax, r9
    ret
sal1_16 ENDP

sal1_32 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0D1h, 0F0h                   ; sal eax, 1
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov QWORD PTR [r8+16], rax
    ret
sal1_32 ENDP

sal1_64 PROC FRAME
    .endprolog
    mov r10, rdx
    db 049h, 0D1h, 0F2h             ; sal r10, 1
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov QWORD PTR [r8+16], r10
    mov rax, r10
    ret
sal1_64 ENDP

sal1_32_imm1 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0C1h, 0F0h, 021h             ; sal eax, 33 (masks to 1)
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov QWORD PTR [r8+16], rax
    ret
sal1_32_imm1 ENDP

sal1_16m PROC FRAME
    .endprolog
    mov QWORD PTR [r8+16], rdx
    db 066h, 041h, 0D1h, 070h, 010h ; sal WORD PTR [r8+16], 1
    seto BYTE PTR [r8]
    setno BYTE PTR [r8+1]
    setb BYTE PTR [r8+2]
    setae BYTE PTR [r8+3]
    sete BYTE PTR [r8+4]
    setne BYTE PTR [r8+5]
    setbe BYTE PTR [r8+6]
    seta BYTE PTR [r8+7]
    sets BYTE PTR [r8+8]
    setns BYTE PTR [r8+9]
    setp BYTE PTR [r8+10]
    setnp BYTE PTR [r8+11]
    setl BYTE PTR [r8+12]
    setge BYTE PTR [r8+13]
    setle BYTE PTR [r8+14]
    setg BYTE PTR [r8+15]
    mov rax, rcx
    ret
sal1_16m ENDP

; --- Flag-producing SAL with count>1 (10 OF-free SETcc conditions across all four widths) ---

saln_8 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0C0h, 0F0h, 003h             ; sal al, 3
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov QWORD PTR [r8+16], rax
    ret
saln_8 ENDP

saln_16 PROC FRAME
    .endprolog
    mov QWORD PTR [r8+16], rdx
    db 066h, 041h, 0C1h, 070h, 010h, 009h ; sal WORD PTR [r8+16], 9
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov rax, rcx
    ret
saln_16 ENDP

saln_32 PROC FRAME
    .endprolog
    mov rax, rdx
    db 0C1h, 0F0h, 01Fh             ; sal eax, 31
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov QWORD PTR [r8+16], rax
    ret
saln_32 ENDP

saln_64 PROC FRAME
    .endprolog
    mov QWORD PTR [r8+16], rdx
    db 049h, 0C1h, 070h, 010h, 00Dh ; sal QWORD PTR [r8+16], 13
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov rax, rcx
    ret
saln_64 ENDP

salw_64 PROC FRAME
    .endprolog
    mov r10, rdx
    db 049h, 0C1h, 0F2h, 03Fh       ; sal r10, 63
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov QWORD PTR [r8+16], r10
    mov rax, r10
    ret
salw_64 ENDP

saln_32m PROC FRAME
    .endprolog
    mov QWORD PTR [r8+16], rdx
    db 041h, 0C1h, 070h, 010h, 005h ; sal DWORD PTR [r8+16], 5
    setb BYTE PTR [r8]
    setae BYTE PTR [r8+1]
    sete BYTE PTR [r8+2]
    setne BYTE PTR [r8+3]
    setbe BYTE PTR [r8+4]
    seta BYTE PTR [r8+5]
    sets BYTE PTR [r8+6]
    setns BYTE PTR [r8+7]
    setp BYTE PTR [r8+8]
    setnp BYTE PTR [r8+9]
    mov rax, rcx
    ret
saln_32m ENDP

; --- Jcc, CMOVcc, narrow return, and nonvolatile/stack frame ---

jcc_sal64_o PROC FRAME
    .endprolog
    mov rax, rcx
    db 048h, 0D1h, 0F0h             ; sal rax, 1
    jo jcc_sal64_o_taken
    ret
jcc_sal64_o_taken:
    not rax
    ret
jcc_sal64_o ENDP

jcc_sal8_be PROC FRAME
    .endprolog
    mov rax, rdx
    db 0C0h, 0F0h, 003h             ; sal al, 3
    jbe jcc_sal8_be_taken
    ret
jcc_sal8_be_taken:
    not rax
    ret
jcc_sal8_be ENDP

cmov_sal PROC FRAME
    .endprolog
    mov rax, rcx
    db 0D1h, 0F0h                   ; sal eax, 1
    mov r9, rdx
    cmovl rax, r9
    mov QWORD PTR [r8], rax
    mov QWORD PTR [r8+16], rdx
    db 049h, 0C1h, 070h, 010h, 00Dh ; sal QWORD PTR [r8+16], 13
    mov rax, rcx
    cmovnp rax, QWORD PTR [r8+16]
    ret
cmov_sal ENDP

sal_narrow_ret PROC FRAME
    .endprolog
    mov al, dl
    db 0D2h, 0F0h                   ; sal al, cl
    ret
sal_narrow_ret ENDP

sal_nonvolatile_frame PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog
    mov rbx, rdx
    db 0D3h, 0F3h                   ; sal ebx, cl
    mov DWORD PTR [rsp], edx
    db 0C1h, 034h, 024h, 005h       ; sal DWORD PTR [rsp], 5
    mov eax, DWORD PTR [rsp]
    mov QWORD PTR [r8], rbx
    mov QWORD PTR [r8+8], rax
    mov QWORD PTR [r8+16], rdx
    mov rax, rbx
    add rsp, 32
    pop rbx
    ret
sal_nonvolatile_frame ENDP

; --- Negative controls (untranslated by CMake; tested directly for rejection) ---

neg_sal_cl_setcc PROC FRAME
    .endprolog
    mov eax, edx
    db 0D3h, 0F0h                   ; sal eax, cl
    setb al
    ret
neg_sal_cl_setcc ENDP

neg_cmp_sal_cl_setcc PROC FRAME
    .endprolog
    mov eax, edx
    cmp ecx, edx
    db 0D3h, 0F0h                   ; sal eax, cl
    setb al
    ret
neg_cmp_sal_cl_setcc ENDP

neg_sal_zero_setcc PROC FRAME
    .endprolog
    mov eax, edx
    db 0C1h, 0F0h, 020h             ; sal eax, 32
    sete al
    ret
neg_sal_zero_setcc ENDP

neg_sal8_wide_setcc PROC FRAME
    .endprolog
    mov eax, edx
    db 0C0h, 0F0h, 008h             ; sal al, 8
    setb al
    ret
neg_sal8_wide_setcc ENDP

neg_sal16_wide_cmov PROC FRAME
    .endprolog
    mov eax, edx
    db 066h, 0C1h, 0F0h, 010h       ; sal ax, 16
    cmove eax, ecx
    ret
neg_sal16_wide_cmov ENDP

neg_saln_seto PROC FRAME
    .endprolog
    mov eax, edx
    db 0C1h, 0F0h, 002h             ; sal eax, 2
    seto al
    ret
neg_saln_seto ENDP

neg_saln_cmovg PROC FRAME
    .endprolog
    mov rax, rcx
    db 048h, 0C1h, 0F2h, 004h       ; sal rdx, 4
    cmovg rax, rdx
    ret
neg_saln_cmovg ENDP

neg_saln_jl PROC FRAME
    .endprolog
    mov rax, rdx
    db 0C1h, 0F0h, 003h             ; sal eax, 3
    jl neg_saln_jl_taken
    ret
neg_saln_jl_taken:
    not rax
    ret
neg_saln_jl ENDP

neg_sal_ah PROC FRAME
    .endprolog
    mov rax, rcx
    db 0D0h, 0F4h                   ; sal ah, 1
    ret
neg_sal_ah ENDP

neg_sal_undef PROC FRAME
    .endprolog
    db 041h, 0D1h, 0F2h             ; sal r10d, 1
    mov rax, r10
    ret
neg_sal_undef ENDP

neg_sal_rsp PROC FRAME
    .endprolog
    db 048h, 0D1h, 0F4h             ; sal rsp, 1
    ret
neg_sal_rsp ENDP

END
