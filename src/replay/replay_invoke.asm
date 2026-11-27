; Invoke a stand-in slot using raw Windows x64 integer/pointer arguments.
; RCX = target, RDX = slot's RCX (receiver or return buffer), R8 = 11 words.
; The recording has no saved XMM registers. This bridge passes only the
; captured integer registers and stack words, and returns the raw RAX.
PUBLIC x2w_replay_invoke
.code
x2w_replay_invoke PROC FRAME
    ; 32 bytes of home space, eight stack arguments, and 8 for alignment.
    sub rsp, 68h
    .allocstack 68h
    .endprolog

    mov r11, rcx
    mov rcx, rdx
    mov r10, r8
    mov rdx, [r10]
    mov r8, [r10+8]
    mov r9, [r10+16]

    stack_offset = 0
    REPT 8
        mov rax, [r10+24+stack_offset]
        mov [rsp+32+stack_offset], rax
        stack_offset = stack_offset + 8
    ENDM

    call r11
    add rsp, 68h
    ret
x2w_replay_invoke ENDP
END
