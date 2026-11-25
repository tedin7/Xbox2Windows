; coverage: record cpu_aot_bmi2_shift
; Source-built MASM fixture for the four flagless BMI2 shift/rotate forms from
; docs/coverage/x64_zen2.tsv group BMI/ADX: RORX, SARX, SHLX, and SHRX in 32-bit
; and 64-bit register and flat [rcx|rdx|r8 + disp] memory forms, plus CMP -> BMI2 ->
; SETcc flag-preservation checks (CF, ZF, SF, OF, PF).
.code

; --- RORX (32-bit and 64-bit register forms, including imm8 masking >= 32/64) ---
op_rorx32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_SHIFT
    rorx eax, ecx, 7
ELSE
    rorx eax, ecx, 13
ENDIF
    ret
op_rorx32 ENDP

op_rorx32_imm32 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 32
    ret
op_rorx32_imm32 ENDP

op_rorx32_imm37 PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, ecx, 37
    ret
op_rorx32_imm37 ENDP

op_rorx64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_SHIFT
    rorx rax, rcx, 11
ELSE
    rorx rax, rcx, 19
ENDIF
    ret
op_rorx64 ENDP

op_rorx64_imm64 PROC FRAME
    .endprolog
    rorx rax, rcx, 64
    ret
op_rorx64_imm64 ENDP

op_rorx64_imm73 PROC FRAME
    .endprolog
    rorx rax, rcx, 73
    ret
op_rorx64_imm73 ENDP

; --- SHLX / SHRX / SARX (32-bit and 64-bit register forms) ---
op_shlx32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_SHIFT
    shrx eax, ecx, edx
ELSE
    shlx eax, ecx, edx
ENDIF
    ret
op_shlx32 ENDP

op_shlx64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_SHIFT
    shrx rax, rcx, rdx
ELSE
    shlx rax, rcx, rdx
ENDIF
    ret
op_shlx64 ENDP

op_shrx32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_SHIFT
    sarx eax, ecx, edx
ELSE
    shrx eax, ecx, edx
ENDIF
    ret
op_shrx32 ENDP

op_shrx64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_SHIFT
    sarx rax, rcx, rdx
ELSE
    shrx rax, rcx, rdx
ENDIF
    ret
op_shrx64 ENDP

op_sarx32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_SHIFT
    shrx eax, ecx, edx
ELSE
    sarx eax, ecx, edx
ENDIF
    ret
op_sarx32 ENDP

op_sarx64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_SHIFT
    shrx rax, rcx, rdx
ELSE
    sarx rax, rcx, rdx
ENDIF
    ret
op_sarx64 ENDP

; --- Flat [rcx|rdx|r8 + disp] m32 / m64 source forms ---
op_rorx32_mem PROC FRAME
    .endprolog
    mov rax, -1
    rorx eax, dword ptr [rcx+4], 9
    ret
op_rorx32_mem ENDP

op_rorx64_mem PROC FRAME
    .endprolog
    rorx rax, qword ptr [rcx+8], 23
    ret
op_rorx64_mem ENDP

op_shlx32_mem PROC FRAME
    .endprolog
    mov rax, -1
    mov r9d, dword ptr [rdx+4]
    shlx eax, dword ptr [rcx+4], r9d
    ret
op_shlx32_mem ENDP

op_shlx64_mem PROC FRAME
    .endprolog
    mov r9, qword ptr [rdx+8]
    shlx rax, qword ptr [rcx+8], r9
    ret
op_shlx64_mem ENDP

op_shrx32_mem PROC FRAME
    .endprolog
    mov rax, -1
    mov r9d, dword ptr [rdx]
    shrx eax, dword ptr [rcx+8], r9d
    ret
op_shrx32_mem ENDP

op_shrx64_mem PROC FRAME
    .endprolog
    mov r9, qword ptr [rdx]
    shrx rax, qword ptr [rcx+16], r9
    ret
op_shrx64_mem ENDP

op_sarx32_mem PROC FRAME
    .endprolog
    mov rax, -1
    mov r9d, dword ptr [rdx+8]
    sarx eax, dword ptr [rcx+12], r9d
    ret
op_sarx32_mem ENDP

op_sarx64_mem PROC FRAME
    .endprolog
    mov r9, qword ptr [rdx+16]
    sarx rax, qword ptr [rcx+24], r9
    ret
op_sarx64_mem ENDP

; Store to [r8] then reload as m64 BMI2 source (tests [r8] base and program order).
op_mem_r8_source PROC FRAME
    .endprolog
    mov r9, qword ptr [rdx]
    rorx r10, qword ptr [rcx], 17
    mov qword ptr [r8], r10
    sarx rax, qword ptr [r8], r9
    ret
op_mem_r8_source ENDP

; --- Flag preservation: CMP -> RORX/SHLX/SHRX/SARX -> SETcc (CF/ZF/SF/OF/PF) ---
op_flags_preserve32 PROC FRAME
    .endprolog
    cmp ecx, edx
    mov rax, -1
    rorx r9d, ecx, 5
    shlx r10d, r9d, edx
    shrx r11d, r10d, edx
    sarx eax, r11d, edx
    setb byte ptr [r8]
    sete byte ptr [r8+1]
    sets byte ptr [r8+2]
    seto byte ptr [r8+3]
    setp byte ptr [r8+4]
    movzx r9d, byte ptr [r8]
    movzx r10d, byte ptr [r8+1]
    mov r11d, 1
    shlx r10d, r10d, r11d
    or r9d, r10d
    movzx r10d, byte ptr [r8+2]
    mov r11d, 2
    shlx r10d, r10d, r11d
    or r9d, r10d
    movzx r10d, byte ptr [r8+3]
    mov r11d, 3
    shlx r10d, r10d, r11d
    or r9d, r10d
    movzx r10d, byte ptr [r8+4]
    mov r11d, 4
    shlx r10d, r10d, r11d
    or r9d, r10d
    mov qword ptr [r8+8], r9
    ret
op_flags_preserve32 ENDP

op_flags_preserve64 PROC FRAME
    .endprolog
    cmp rcx, rdx
    rorx r9, rcx, 11
    shlx r10, r9, rdx
    shrx r11, r10, rdx
    sarx rax, r11, rdx
    setb byte ptr [r8]
    sete byte ptr [r8+1]
    sets byte ptr [r8+2]
    seto byte ptr [r8+3]
    setp byte ptr [r8+4]
    movzx r9, byte ptr [r8]
    movzx r10, byte ptr [r8+1]
    mov r11, 1
    shlx r10, r10, r11
    or r9, r10
    movzx r10, byte ptr [r8+2]
    mov r11, 2
    shlx r10, r10, r11
    or r9, r10
    movzx r10, byte ptr [r8+3]
    mov r11, 3
    shlx r10, r10, r11
    or r9, r10
    movzx r10, byte ptr [r8+4]
    mov r11, 4
    shlx r10, r10, r11
    or r9, r10
    mov qword ptr [r8+8], r9
    ret
op_flags_preserve64 ENDP

op_flags_preserve_mem PROC FRAME
    .endprolog
    mov r9, qword ptr [rcx]
    mov r10, qword ptr [rdx]
    cmp r9, r10
    rorx r11, qword ptr [rcx+8], 7
    shlx r11, qword ptr [rcx+16], r10
    shrx r11, qword ptr [rcx+24], r10
    sarx rax, qword ptr [rcx+8], r10
    setb byte ptr [r8]
    sete byte ptr [r8+1]
    sets byte ptr [r8+2]
    seto byte ptr [r8+3]
    setp byte ptr [r8+4]
    ret
op_flags_preserve_mem ENDP

; --- Discarded memory read must still fault on NULL rdx even under /O2 ---
op_bmi2_discarded_read PROC FRAME
    .endprolog
    rorx r9, qword ptr [rdx], 7
    mov rax, rcx
    ret
op_bmi2_discarded_read ENDP

END
