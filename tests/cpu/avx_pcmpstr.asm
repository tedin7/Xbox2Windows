; coverage: source cpu_aot_avx_pcmpstr
; Source-built MASM fixture exercising VEX.128 AVX packed string compare forms
; (vpcmpestri, vpcmpestrm, vpcmpistri, vpcmpistrm) from docs/coverage/x64_zen2.tsv group AVX.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes; [0..15] string A, [16..31] YMM sentinel)
;   RDX = const void* input_b (16 bytes; string B)
;   R8  = void*       io      (64 bytes; [0..3] la / [4..7] lb on input for est*;
;                              on output [0..7] RCX index for *stri, [0..31] YMM0 for *strm,
;                              [32..47] staged m128 or preserved XMM1)
; The VEX.W=1 (64-bit length) exports read signed int64 la / lb from [R8+48] / [R8+56] instead.
; Return value in RAX: (CF << 0) | (ZF << 1) | (SF << 2) | (OF << 3) | (PF << 4)

; VEX.128 W=1 64-bit explicit-length compares, emitted as raw bytes.
VPCMPESTRI64_XMM1_XMM2 MACRO imm
    DB 0C4h, 0E3h, 0F9h, 61h, 0CAh, imm
ENDM

VPCMPESTRI64_XMM1_R8_32 MACRO imm
    DB 0C4h, 0C3h, 0F9h, 61h, 48h, 20h, imm
ENDM

VPCMPESTRM64_XMM1_XMM2 MACRO imm
    DB 0C4h, 0E3h, 0F9h, 60h, 0CAh, imm
ENDM

PACK_FLAGS MACRO
    setb al
    sete r9b
    sets r10b
    seto r11b
    setp cl
    movzx eax, al
    movzx r9d, r9b
    shl r9d, 1
    or eax, r9d
    movzx r10d, r10b
    shl r10d, 2
    or eax, r10d
    movzx r11d, r11b
    shl r11d, 3
    or eax, r11d
    movzx ecx, cl
    shl ecx, 4
    or eax, ecx
    ret
ENDM

.code

; --- VPCMPESTRI (explicit length, index in ECX -> zero-extended RCX) ---

vpcmpestri_ub_eqany_pos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, DWORD PTR [r8]
    or rax, r9
    mov edx, DWORD PTR [r8+4]
    or rdx, r9
    mov r10d, -1
    or r10d, r10d
    mov rcx, -1
IFDEF ALTER_IMM
    vpcmpestri xmm1, xmm2, 01h
ELSE
    vpcmpestri xmm1, xmm2, 00h
ENDIF
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_ub_eqany_pos_lsb ENDP

vpcmpestri_sw_ranges_neg_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, DWORD PTR [r8]
    or rax, r9
    mov edx, DWORD PTR [r8+4]
    or rdx, r9
    mov rcx, -1
    vpcmpestri xmm1, xmm2, 57h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_sw_ranges_neg_msb ENDP

vpcmpestri_sb_eqeach_mpos_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    mov rcx, -1
    vpcmpestri xmm1, xmm2, 6Ah
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_sb_eqeach_mpos_msb ENDP

vpcmpestri_uw_eqord_mneg_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    mov rcx, -1
    vpcmpestri xmm1, xmm2, 3Dh
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_uw_eqord_mneg_lsb ENDP

vpcmpestri_mem_ub_eqany_pos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu XMMWORD PTR [r8+32], xmm2
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    mov rcx, -1
    vpcmpestri xmm1, XMMWORD PTR [r8+32], 00h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_mem_ub_eqany_pos_lsb ENDP

vpcmpestri_alias_xmm0 PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    mov rcx, -1
    vpcmpestri xmm0, xmm0, 1Ch
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri_alias_xmm0 ENDP

; --- VPCMPESTRM (explicit length, mask in XMM0, zeroes YMM0[255:128]) ---

vpcmpestrm_ub_eqany_pos_bit PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, DWORD PTR [r8]
    or rax, r9
    mov edx, DWORD PTR [r8+4]
    or rdx, r9
    mov r10d, -1
    or r10d, r10d
    vpcmpestrm xmm1, xmm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu XMMWORD PTR [r8+32], xmm1
    vzeroupper
    PACK_FLAGS
vpcmpestrm_ub_eqany_pos_bit ENDP

vpcmpestrm_sb_ranges_neg_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
IFDEF ALTER_IMM
    vpcmpestrm xmm1, xmm2, 16h
ELSE
    vpcmpestrm xmm1, xmm2, 56h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpestrm_sb_ranges_neg_exp ENDP

vpcmpestrm_uw_eqeach_mpos_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    vpcmpestrm xmm1, xmm2, 69h
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu XMMWORD PTR [r8+32], xmm1
    vzeroupper
    PACK_FLAGS
vpcmpestrm_uw_eqeach_mpos_exp ENDP

vpcmpestrm_sw_eqord_mneg_bit PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    vpcmpestrm xmm1, xmm2, 3Fh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpestrm_sw_eqord_mneg_bit ENDP

vpcmpestrm_mem_ub_eqany_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu XMMWORD PTR [r8+32], xmm2
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    vpcmpestrm xmm1, XMMWORD PTR [r8+32], 40h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpestrm_mem_ub_eqany_exp ENDP

vpcmpestrm_xmm0_self PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
    mov eax, DWORD PTR [r8]
    mov edx, DWORD PTR [r8+4]
    vpcmpestrm xmm0, xmm1, 40h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpestrm_xmm0_self ENDP

; --- VPCMPISTRI (implicit NUL-terminated length, index in ECX -> zero-extended RCX) ---

vpcmpistri_ub_eqord_pos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    xor r10d, r10d
    mov rcx, -1
IFDEF ALTER_IMM
    vpcmpistri xmm1, xmm2, 0CCh
ELSE
    vpcmpistri xmm1, xmm2, 8Ch
ENDIF
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpistri_ub_eqord_pos_lsb ENDP

vpcmpistri_sb_ranges_mneg_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rcx, -1
    vpcmpistri xmm1, xmm2, 76h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpistri_sb_ranges_mneg_msb ENDP

vpcmpistri_uw_eqany_neg_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rcx, -1
    vpcmpistri xmm1, xmm2, 51h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpistri_uw_eqany_neg_msb ENDP

vpcmpistri_sw_eqeach_mpos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rcx, -1
    vpcmpistri xmm1, xmm2, 2Bh
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpistri_sw_eqeach_mpos_lsb ENDP

vpcmpistri_mem_ub_eqany_pos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov rcx, -1
    vpcmpistri xmm1, XMMWORD PTR [rdx], 00h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpistri_mem_ub_eqany_pos_lsb ENDP

; --- VPCMPISTRM (implicit NUL-terminated length, mask in XMM0, zeroes YMM0[255:128]) ---

vpcmpistrm_ub_eqeach_pos_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpistrm xmm1, xmm2, 48h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpistrm_ub_eqeach_pos_exp ENDP

vpcmpistrm_sw_eqany_mneg_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov r10d, -1
    or r10d, r10d
IFDEF ALTER_IMM
    vpcmpistrm xmm1, xmm2, 33h
ELSE
    vpcmpistrm xmm1, xmm2, 73h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu XMMWORD PTR [r8+32], xmm1
    vzeroupper
    PACK_FLAGS
vpcmpistrm_sw_eqany_mneg_exp ENDP

vpcmpistrm_uw_ranges_mpos_bit PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpistrm xmm1, xmm2, 25h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpistrm_uw_ranges_mpos_bit ENDP

vpcmpistrm_sb_eqord_neg_bit PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpistrm xmm1, xmm2, 1Eh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpistrm_sb_eqord_neg_bit ENDP

vpcmpistrm_mem_ub_eqord_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpcmpistrm xmm1, XMMWORD PTR [rdx], 4Ch
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    PACK_FLAGS
vpcmpistrm_mem_ub_eqord_exp ENDP

; --- VPCMPESTRI64 / VPCMPESTRM64 (VEX.W=1, full signed 64-bit RAX/RDX lengths) ---

vpcmpestri64_ub_eqany_pos_lsb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rax, QWORD PTR [r8+48]
    mov rdx, QWORD PTR [r8+56]
    mov rcx, -1
IFDEF ALTER_IMM
    VPCMPESTRI64_XMM1_XMM2 01h
ELSE
    VPCMPESTRI64_XMM1_XMM2 00h
ENDIF
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri64_ub_eqany_pos_lsb ENDP

vpcmpestri64_sb_eqeach_mpos_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rax, QWORD PTR [r8+48]
    mov rdx, QWORD PTR [r8+56]
    mov rcx, -1
    VPCMPESTRI64_XMM1_XMM2 6Ah
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri64_sb_eqeach_mpos_msb ENDP

vpcmpestri64_mem_sw_ranges_neg_msb PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu XMMWORD PTR [r8+32], xmm2
    mov rax, QWORD PTR [r8+48]
    mov rdx, QWORD PTR [r8+56]
    mov rcx, -1
    VPCMPESTRI64_XMM1_R8_32 57h
    mov QWORD PTR [r8], rcx
    vzeroupper
    PACK_FLAGS
vpcmpestri64_mem_sw_ranges_neg_msb ENDP

vpcmpestrm64_ub_eqany_pos_bit PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rax, QWORD PTR [r8+48]
    mov rdx, QWORD PTR [r8+56]
    VPCMPESTRM64_XMM1_XMM2 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu XMMWORD PTR [r8+32], xmm1
    vzeroupper
    PACK_FLAGS
vpcmpestrm64_ub_eqany_pos_bit ENDP

vpcmpestrm64_uw_eqeach_mpos_exp PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [r8]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    mov rax, QWORD PTR [r8+48]
    mov rdx, QWORD PTR [r8+56]
IFDEF ALTER_IMM
    VPCMPESTRM64_XMM1_XMM2 29h
ELSE
    VPCMPESTRM64_XMM1_XMM2 69h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu XMMWORD PTR [r8+32], xmm1
    vzeroupper
    PACK_FLAGS
vpcmpestrm64_uw_eqeach_mpos_exp ENDP

vpcmpistrm_discarded_read PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov eax, 11223344h
    mov DWORD PTR [r8], eax
    vpcmpistrm xmm1, XMMWORD PTR [rdx], 00h
    mov eax, 55667788h
    mov DWORD PTR [r8], eax
    ret
vpcmpistrm_discarded_read ENDP

END
