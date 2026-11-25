; coverage: source cpu_aot_switch_bit_string
; Source-built register-indexed memory bit tests.
.data
bit_string_data QWORD 0
.code

bt16_string PROC FRAME
    .endprolog
    xor eax, eax
    bt WORD PTR [rdx], cx
    setb al
    ret
bt16_string ENDP

bt32_string PROC FRAME
    .endprolog
    xor eax, eax
    bt DWORD PTR [rdx], ecx
    setb al
    ret
bt32_string ENDP

bt64_string PROC FRAME
    .endprolog
    xor eax, eax
    bt QWORD PTR [rdx], rcx
    setb al
    ret
bt64_string ENDP

bts16_string PROC FRAME
    .endprolog
    xor eax, eax
    bts WORD PTR [rdx], cx
    setb al
    ret
bts16_string ENDP

bts32_string PROC FRAME
    .endprolog
    xor eax, eax
    bts DWORD PTR [rdx], ecx
    setb al
    ret
bts32_string ENDP

bts64_string PROC FRAME
    .endprolog
    xor eax, eax
    bts QWORD PTR [rdx], rcx
    setb al
    ret
bts64_string ENDP

btr16_string PROC FRAME
    .endprolog
    xor eax, eax
    btr WORD PTR [rdx], cx
    setb al
    ret
btr16_string ENDP

btr32_string PROC FRAME
    .endprolog
    xor eax, eax
    btr DWORD PTR [rdx], ecx
    setb al
    ret
btr32_string ENDP

btr64_string PROC FRAME
    .endprolog
    xor eax, eax
    btr QWORD PTR [rdx], rcx
    setb al
    ret
btr64_string ENDP

btc16_string PROC FRAME
    .endprolog
    xor eax, eax
    btc WORD PTR [rdx], cx
    setb al
    ret
btc16_string ENDP

btc32_string PROC FRAME
    .endprolog
    xor eax, eax
    btc DWORD PTR [rdx], ecx
    setb al
    ret
btc32_string ENDP

btc64_string PROC FRAME
    .endprolog
    xor eax, eax
    btc QWORD PTR [rdx], rcx
    setb al
    ret
btc64_string ENDP

bt64_dead PROC FRAME
    .endprolog
    bt QWORD PTR [rdx], rcx
    xor eax, eax
    ret
bt64_dead ENDP

btc64_alias PROC FRAME
    .endprolog
    xor eax, eax
    btc QWORD PTR [rdx+rcx], rcx
    setb al
    ret
btc64_alias ENDP

neg_bit_frame PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov QWORD PTR [rsp+10h], rdx
    bt QWORD PTR [rsp+10h], rcx
    add rsp, 28h
    xor eax, eax
    ret
neg_bit_frame ENDP

neg_bit_image PROC FRAME
    .endprolog
    bt QWORD PTR [bit_string_data], rcx
    xor eax, eax
    ret
neg_bit_image ENDP

neg_bit_undef PROC FRAME
    .endprolog
    bt QWORD PTR [rdx], r10
    xor eax, eax
    ret
neg_bit_undef ENDP

END
