; Source-owned differential fixture for recursion through direct calls in the jump-table translator.
; Each export is fn(x, y, trace) with x in RCX, y in RDX and a three-word trace in R8, and returns
; RAX. The recursive functions are not exported: each has its own runtime-function entry. The
; register probe seeds RBX and RSI, so a save or restore that the translation loses shows up as a
; changed sentinel. The exports from tail_ping on exist only to be refused.
.code

; sum_down(n, k): the sum of i XOR k for i = n down to 1, by self-recursion through CALL. RBX and
; RSI keep n and k across the call.
sum_down PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    xor eax, eax
    test rcx, rcx
    jz sum_down_done
    mov rbx, rcx
    mov rsi, rdx
    lea rcx, [rbx-1]
    call sum_down
    mov rcx, rbx
    xor rcx, rsi
    add rax, rcx
sum_down_done:
    add rsp, 28h
    pop rsi
    pop rbx
    ret
sum_down ENDP

; is_even(n): AL = 1 when n is even, through is_odd. Only AL is defined at the base case.
is_even PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test rcx, rcx
    jnz is_even_step
    mov al, 1
    add rsp, 28h
    ret
is_even_step:
    dec rcx
    call is_odd
    add rsp, 28h
    ret
is_even ENDP

; is_odd(n): AL = 1 when n is odd, through is_even. Its base case defines all of RAX, so only
; the cycle limits its return to AL.
is_odd PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test rcx, rcx
    jnz is_odd_step
    xor eax, eax
    add rsp, 28h
    ret
is_odd_step:
    dec rcx
    call is_even
    add rsp, 28h
    ret
is_odd ENDP

; count_down(n): the sum of i AND 3 for i = n down to 1. Its own cycle, reached only from walk_b.
count_down PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    xor eax, eax
    test rcx, rcx
    jz count_down_done
    mov rbx, rcx
    dec rcx
    call count_down
    and ebx, 3
    add rax, rbx
count_down_done:
    add rsp, 20h
    pop rbx
    ret
count_down ENDP

; walk_a(n, v, k): v XOR k when n = 0, otherwise walk_b(n - 1, v + n, k) + 1. The only read of k
; (R8) in the cycle walk_a -> walk_b -> walk_c -> walk_a is here.
walk_a PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test rcx, rcx
    jnz walk_a_step
    mov rax, rdx
    xor rax, r8
    add rsp, 28h
    ret
walk_a_step:
    add rdx, rcx
    dec rcx
    call walk_b
    inc rax
    add rsp, 28h
    ret
walk_a ENDP

; walk_b(n, v, k): v when n = 0, otherwise walk_c(n - 1, 3 * v, k) XOR count_down(n).
walk_b PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rax, rdx
    test rcx, rcx
    jz walk_b_done
    mov rbx, rcx
    lea rdx, [rdx+rdx*2]
    dec rcx
    call walk_c
    mov rsi, rax
    mov rcx, rbx
    call count_down
    xor rax, rsi
walk_b_done:
    add rsp, 28h
    pop rsi
    pop rbx
    ret
walk_b ENDP

; walk_c(n, v, k): v when n = 0, otherwise walk_a(n - 1, v + 5, k). It passes k on without
; reading it, so only the cycle makes it demand R8.
walk_c PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rax, rdx
    test rcx, rcx
    jz walk_c_done
    dec rcx
    add rdx, 5
    call walk_a
walk_c_done:
    add rsp, 28h
    ret
walk_c ENDP

; trace[0] = sum_down(x AND 1FFh, y); returns it plus x.
sum_export PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, r8
    mov rsi, rcx
    and ecx, 1FFh
    call sum_down
    mov qword ptr [rbx], rax
    add rax, rsi
    add rsp, 28h
    pop rsi
    pop rbx
    ret
sum_export ENDP

; trace[1] = is_even(x AND 0FFh), trace[2] = is_odd(y AND 0FFh); returns their sum. Both
; results are AL only, which MOVZX widens.
parity_export PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, r8
    mov rsi, rdx
    movzx ecx, cl
    call is_even
    movzx eax, al
    mov qword ptr [rbx+8], rax
    movzx ecx, sil
    call is_odd
    movzx eax, al
    mov qword ptr [rbx+10h], rax
    add rax, qword ptr [rbx+8]
    add rsp, 28h
    pop rsi
    pop rbx
    ret
parity_export ENDP

; trace[2] = walk_a(x AND 3Fh, y, x XOR y).
walk_export PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, r8
    mov r8, rcx
    xor r8, rdx
    and ecx, 3Fh
    call walk_a
    mov qword ptr [rbx+10h], rax
    add rsp, 20h
    pop rbx
    ret
walk_export ENDP

; fold_export(x, y, trace): y when x = 0, with trace[2] = y; otherwise fold_export(x SHR 1,
; 2 * y + (x AND 1), trace) + (x AND 1). It calls its own exported entry.
fold_export PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    test rcx, rcx
    jnz fold_step
    mov qword ptr [r8+10h], rdx
    mov rax, rdx
    jmp fold_done
fold_step:
    mov ebx, ecx
    and ebx, 1
    lea rdx, [rbx+rdx*2]
    shr rcx, 1
    call fold_export
    add rax, rbx
fold_done:
    add rsp, 20h
    pop rbx
    ret
fold_export ENDP

; Refused: tail_ping and tail_pong form a cycle of tail jumps, which the original runs in
; constant stack.
tail_ping PROC FRAME
    .endprolog
    test rcx, rcx
    jz tail_ping_done
    dec rcx
    jmp tail_pong
tail_ping_done:
    mov rax, rdx
    ret
tail_ping ENDP

tail_pong PROC FRAME
    .endprolog
    add rdx, 3
    jmp tail_ping
tail_pong ENDP

; Refused: after is_even, is_odd returns only AL, so the full read of RAX is undefined.
wide_odd PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, rcx
    call is_even
    mov rcx, rbx
    call is_odd
    add rax, rbx
    add rsp, 20h
    pop rbx
    ret
wide_odd ENDP

; Refused: after walk_a, walk_c is called with R8 undefined, and the cycle reads it.
lost_k PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, rcx
    call walk_a
    mov rcx, rbx
    mov rdx, rax
    call walk_c
    add rsp, 20h
    pop rbx
    ret
lost_k ENDP

; Refused: bad_b, in a cycle with bad_a, executes CPUID, which the translator does not lower.
bad_a PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    xor eax, eax
    test rcx, rcx
    jz bad_a_done
    dec rcx
    call bad_b
bad_a_done:
    add rsp, 28h
    ret
bad_a ENDP

bad_b PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, rcx
    xor eax, eax
    cpuid
    mov rcx, rbx
    call bad_a
    add rsp, 20h
    pop rbx
    ret
bad_b ENDP


; Refused: stale_a reads all of RAX after stale_t, which returns only AL through the cycle
; stale_a -> stale_s -> stale_q -> stale_s -> stale_a. stale_t is translated after stale_s and
; stale_q are pending and reuses stale_q, so it is in the same cycle and must not be kept on its own.
stale_a PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test rcx, rcx
    jnz stale_a_step
    mov al, 1
    add rsp, 28h
    ret
stale_a_step:
    dec rcx
    jz stale_a_t
    call stale_s
    add rsp, 28h
    ret
stale_a_t:
    call stale_t
    inc rax
    add rsp, 28h
    ret
stale_a ENDP

stale_s PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test rcx, rcx
    jz stale_s_a
    call stale_q
    add rsp, 28h
    ret
stale_s_a:
    call stale_a
    add rsp, 28h
    ret
stale_s ENDP

stale_q PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    dec rcx
    call stale_s
    add rsp, 28h
    ret
stale_q ENDP

stale_t PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call stale_q
    add rsp, 28h
    ret
stale_t ENDP

stale_e PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call stale_a
    add rsp, 28h
    ret
stale_e ENDP

END
