; Source-owned differential fixture for stack arguments (the fifth and later) of internal callees
; in the jump-table translator. Each export is fn(x, y, trace) with x in RCX, y in RDX and a
; three-word trace in R8, and returns RAX. The callees are not exported: each has its own
; runtime-function entry. A caller writes the stack arguments above the callee's home space, at
; RSP+20h and up when the CALL executes, and the callee reads them at entry RSP+28h and up. The
; register probe seeds RBX and RSI. The exports from unset_slot on exist only to be refused.
.code

; six_mix(a, b, c, d, e, f) = a + 2b + 3c + 4d + 5e + 6f, with e a quadword and f a doubleword.
six_mix PROC FRAME
    .endprolog
    lea rax, [rcx+rdx*2]
    lea rdx, [r8+r8*2]
    add rax, rdx
    lea rax, [rax+r9*4]
    mov rdx, qword ptr [rsp+28h]
    lea rdx, [rdx+rdx*4]
    add rax, rdx
    mov edx, dword ptr [rsp+30h]
    imul rdx, rdx, 6
    add rax, rdx
    ret
six_mix ENDP

; seven_fwd(a, b, c, d, e, f, g) = six_mix(a, b, c, d, f, e) + 7 * g, with g a byte. RSI keeps g
; across the call, and e and f are forwarded swapped as the callee's fifth and sixth arguments.
seven_fwd PROC FRAME
    push rsi
    .pushreg rsi
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    movzx esi, byte ptr [rsp+70h]
    mov rax, qword ptr [rsp+68h]
    mov qword ptr [rsp+20h], rax
    mov rax, qword ptr [rsp+60h]
    mov dword ptr [rsp+28h], eax
    call six_mix
    imul rsi, rsi, 7
    add rax, rsi
    add rsp, 30h
    pop rsi
    ret
seven_fwd ENDP

; scratch_five(a, b, c, d, e): adds a to its own fifth-argument slot and returns e + a XOR b.
scratch_five PROC FRAME
    .endprolog
    add qword ptr [rsp+28h], rcx
    mov rax, qword ptr [rsp+28h]
    xor rax, rdx
    ret
scratch_five ENDP

; rec_five(n, acc, c, d, step): acc when n = 0, otherwise rec_five(n - 1, acc + step, c, d, 3 * step).
rec_five PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov rax, rdx
    test rcx, rcx
    jz rec_five_done
    mov rax, qword ptr [rsp+60h]
    add rdx, rax
    lea rax, [rax+rax*2]
    mov qword ptr [rsp+20h], rax
    dec rcx
    call rec_five
rec_five_done:
    add rsp, 38h
    ret
rec_five ENDP

; eight_sum(a, ..., h) = a + 3b + 5c + 7d + 9e + 11f + 13g + 15h, h a word.
eight_sum PROC FRAME
    .endprolog
    lea rax, [rdx+rdx*2]
    add rax, rcx
    lea rcx, [r8+r8*4]
    add rax, rcx
    imul rcx, r9, 7
    add rax, rcx
    imul rcx, qword ptr [rsp+28h], 9
    add rax, rcx
    imul rcx, qword ptr [rsp+30h], 11
    add rax, rcx
    imul rcx, qword ptr [rsp+38h], 13
    add rax, rcx
    movzx ecx, word ptr [rsp+40h]
    imul rcx, rcx, 15
    add rax, rcx
    ret
eight_sum ENDP

; fill_five(a, b, c, d, e): writes a to its fifth-argument slot without reading it; returns 3a.
fill_five PROC FRAME
    .endprolog
    mov qword ptr [rsp+28h], rcx
    lea rax, [rcx+rcx*2]
    ret
fill_five ENDP

; twelfth(a, ..., l) = a XOR l: the twelfth argument, in the last supported slot.
twelfth PROC FRAME
    .endprolog
    mov rax, qword ptr [rsp+60h]
    xor rax, rcx
    ret
twelfth ENDP

; rec_lost(n, b, c, d, e): e when n = 0, otherwise rec_lost(n - 1, ...) without writing its fifth
; argument. Refused through lost_slot below.
rec_lost PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov rax, qword ptr [rsp+60h]
    test rcx, rcx
    jz rec_lost_done
    dec rcx
    call rec_lost
rec_lost_done:
    add rsp, 38h
    ret
rec_lost ENDP

; trace[0] = six_mix(x, y, x XOR y, y + 1, x, y AND 0FFFFFFFFh); returns it. Only the low
; doubleword of the sixth argument is written.
call_six PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    mov rbx, r8
    mov qword ptr [rsp+20h], rcx
    mov dword ptr [rsp+28h], edx
    mov r8, rcx
    xor r8, rdx
    lea r9, [rdx+1]
    call six_mix
    mov qword ptr [rbx], rax
    add rsp, 30h
    pop rbx
    ret
call_six ENDP

; trace[1] = seven_fwd(y, x, 5, x + y, y, x, x SHR 8 AND 0FFh); returns it plus y. The seventh
; argument is one byte, written over a slot that held an earlier quadword.
call_seven PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov rbx, r8
    mov rsi, rdx
    mov qword ptr [rsp+30h], rcx
    mov qword ptr [rsp+20h], rdx
    mov qword ptr [rsp+28h], rcx
    mov rax, rcx
    shr rax, 8
    mov byte ptr [rsp+30h], al
    lea r9, [rcx+rdx]
    mov r8d, 5
    mov rdx, rcx
    mov rcx, rsi
    call seven_fwd
    mov qword ptr [rbx+8], rax
    add rax, rsi
    add rsp, 38h
    pop rsi
    pop rbx
    ret
call_seven ENDP

; trace[2] = scratch_five(x, y, 0, 0, y * 2); returns it XOR x. The slot the callee modifies is
; written again before a second call, which uses x as both a and e.
call_scratch PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, r8
    mov rsi, rcx
    lea rax, [rdx+rdx]
    mov qword ptr [rsp+20h], rax
    xor r8d, r8d
    xor r9d, r9d
    call scratch_five
    mov qword ptr [rbx+10h], rax
    mov qword ptr [rsp+20h], rsi
    mov rcx, rsi
    mov rdx, rax
    xor r8d, r8d
    xor r9d, r9d
    call scratch_five
    xor rax, rsi
    add rsp, 28h
    pop rsi
    pop rbx
    ret
call_scratch ENDP

; trace[0] = rec_five(x AND 1Fh, y, 0, 0, x); returns it.
call_rec PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    mov rbx, r8
    mov qword ptr [rsp+20h], rcx
    and ecx, 1Fh
    xor r8d, r8d
    xor r9d, r9d
    call rec_five
    mov qword ptr [rbx], rax
    add rsp, 30h
    pop rbx
    ret
call_rec ENDP

; trace[1] = eight_sum(x, y, x + y, x - y, y, x, x XOR y, y AND 0FFFFh); returns it.
call_eight PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 40h
    .allocstack 40h
    .endprolog
    mov rbx, r8
    mov qword ptr [rsp+20h], rdx
    mov qword ptr [rsp+28h], rcx
    mov rax, rcx
    xor rax, rdx
    mov qword ptr [rsp+30h], rax
    mov word ptr [rsp+38h], dx
    lea r8, [rcx+rdx]
    mov r9, rcx
    sub r9, rdx
    call eight_sum
    mov qword ptr [rbx+8], rax
    add rsp, 40h
    pop rbx
    ret
call_eight ENDP

; Returns fill_five(x, ...) = 3x. fill_five reads none of its stack arguments, so this caller
; writes no stack byte, and its one slot ends exactly at this function's return address.
call_fill PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call fill_five
    add rsp, 28h
    ret
call_fill ENDP

; Returns scratch_five(y, x, 0, 0, x) = (x + y) XOR x, with its slot ending exactly at the return address.
call_edge PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov qword ptr [rsp+20h], rcx
    mov rax, rcx
    mov rcx, rdx
    mov rdx, rax
    xor r8d, r8d
    xor r9d, r9d
    call scratch_five
    add rsp, 28h
    ret
call_edge ENDP

; Returns twelfth(x, ..., y) = x XOR y; only the twelfth argument is written.
call_twelfth PROC FRAME
    sub rsp, 68h
    .allocstack 68h
    .endprolog
    mov qword ptr [rsp+58h], rdx
    call twelfth
    add rsp, 68h
    ret
call_twelfth ENDP

; Refused: rec_lost calls itself without writing the fifth argument it reads.
lost_slot PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov qword ptr [rsp+20h], rdx
    xor r9d, r9d
    call rec_lost
    add rsp, 38h
    ret
lost_slot ENDP

; Refused: the sixth argument of six_mix is never written.
unset_slot PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov qword ptr [rsp+20h], rcx
    xor r9d, r9d
    call six_mix
    add rsp, 38h
    ret
unset_slot ENDP

; Refused: only the low doubleword of the fifth argument is written, and six_mix reads all 8 bytes.
narrow_slot PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    mov dword ptr [rsp+28h], edx
    xor r9d, r9d
    call six_mix
    add rsp, 38h
    ret
narrow_slot ENDP

; Refused: scratch_five may change its stack argument, so the slot is undefined after the call.
reread_slot PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov qword ptr [rsp+20h], rcx
    call scratch_five
    add rax, qword ptr [rsp+20h]
    add rsp, 38h
    ret
reread_slot ENDP

; Refused: a tail jump to six_mix would pass this function's own stack arguments.
tail_six PROC FRAME
    .endprolog
    jmp six_mix
tail_six ENDP

; Refused: an export reads a fifth argument, which its host call does not pass.
export_fifth PROC FRAME
    .endprolog
    mov rax, qword ptr [rsp+28h]
    ret
export_fifth ENDP

; Refused: far_ninth reads the thirteenth argument, above the eight stack arguments supported.
far_slot PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call far_ninth
    add rsp, 28h
    ret
far_slot ENDP

far_ninth PROC FRAME
    .endprolog
    mov rax, qword ptr [rsp+68h]
    ret
far_ninth ENDP

; Refused: straddle reads 8 bytes across the end of its home space and the fifth argument.
straddle_slot PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call straddle
    add rsp, 28h
    ret
straddle_slot ENDP

straddle PROC FRAME
    .endprolog
    mov rax, qword ptr [rsp+24h]
    ret
straddle ENDP

; Refused: with RSP at entry RSP - 28h, the two stack arguments of six_mix would be this
; function's return address and the slot above it.
high_slots PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call six_mix
    add rsp, 28h
    ret
high_slots ENDP

END
