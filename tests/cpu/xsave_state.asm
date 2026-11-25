; coverage: source cpu_aot_xsave_state
; Source-built MASM fixture exercising 8 XSAVE state forms from docs/coverage/x64_zen2.tsv
; group XSAVE: xsave, xsave64, xsavec, xsavec64, xsaveopt, xsaveopt64, xrstor, and xrstor64 (standard and compacted
; format), along with transport/ABI glue (vmovdqu, vzeroupper, paddd, movq, emms).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (64-byte aligned; 960+ bytes)
;   RDX = const void* input_b (64-byte aligned; 960+ bytes)
;   R8  = void*       output  (64-byte aligned; 960+ bytes)
.code

; --- Entry-state XSAVE / XSAVE64 (opt-in canonical-entry profile) ---
; Native callers targeting lifted canonical exports must establish canonical XSAVE state
; (FCW=0x037F, FSW=0, FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0, MXCSR=0x1F80, zero ST/XMM/YMM_HI, XINUSE=0).

op_xsave_only PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xsave [r8]
    ret
op_xsave_only ENDP

op_xsave64_only PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xsave64 [r8]
    ret
op_xsave64_only ENDP

; --- XRSTOR / XRSTOR64 -> XSAVE / XSAVE64 roundtrips ---

op_xrstor_xsave PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor [rcx]
    xsave [r8]
    ret
op_xrstor_xsave ENDP

op_xrstor64_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    xsave [r8]
ELSE
    xsave64 [r8]
ENDIF
    ret
op_xrstor64_xsave64 ENDP

op_xrstor_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
IFDEF ALTER_OP
    xrstor64 [rcx]
ELSE
    xrstor [rcx]
ENDIF
    xsave64 [r8]
    ret
op_xrstor_xsave64 ENDP

op_xrstor64_xsave PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsave [r8]
    ret
op_xrstor64_xsave ENDP

op_xrstor64_disp PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
IFDEF ALTER_OP
    xrstor [rcx+64]
ELSE
    xrstor64 [rcx+64]
ENDIF
    xsave64 [r8+64]
    ret
op_xrstor64_disp ENDP

op_xrstor64_mask_sse PROC FRAME
    .endprolog
    mov eax, 2
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    mov eax, 7
ELSE
    mov eax, 2
ENDIF
    xsave64 [r8]
    ret
op_xrstor64_mask_sse ENDP

op_vzeroupper_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    vzeroupper
    xsave64 [r8]
    ret
op_vzeroupper_xsave64 ENDP

; --- XSAVEC / XSAVEC64 (compacted format) after a full XRSTOR64 ---

op_xrstor64_xsavec PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsavec [r8]
    ret
op_xrstor64_xsavec ENDP

op_xrstor64_xsavec64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    xsave64 [r8]
ELSE
    xsavec64 [r8]
ENDIF
    ret
op_xrstor64_xsavec64 ENDP

op_xrstor64_xsavec64_sse PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    mov eax, 7
ELSE
    mov eax, 2
ENDIF
    xsavec64 [r8]
    ret
op_xrstor64_xsavec64_sse ENDP

; RFBM = x87|AVX leaves the SSE slot out of the request; AVX stays at offset 576.
op_xrstor64_xsavec64_x87_avx PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov eax, 5
    xsavec64 [r8]
    ret
op_xrstor64_xsavec64_x87_avx ENDP

; The request mask comes from input_b (RAX = [rdx], RDX = [rdx+8]).
op_xsavec64_dynmask PROC FRAME
    .endprolog
    mov r9, rdx
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov rax, QWORD PTR [r9]
    mov rdx, QWORD PTR [r9+8]
    xsavec64 [r8]
    ret
op_xsavec64_dynmask ENDP

; --- XSAVEOPT / XSAVEOPT64 (standard format, init optimization) after a full XRSTOR64 ---
; The restore reads input_a and the save targets output, so the modified optimization never applies.

op_xrstor64_xsaveopt PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsaveopt [r8]
    ret
op_xrstor64_xsaveopt ENDP

op_xrstor64_xsaveopt64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    xsave64 [r8]
ELSE
    xsaveopt64 [r8]
ENDIF
    ret
op_xrstor64_xsaveopt64 ENDP

op_xrstor64_xsaveopt64_sse PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    mov eax, 7
ELSE
    mov eax, 2
ENDIF
    xsaveopt64 [r8]
    ret
op_xrstor64_xsaveopt64_sse ENDP

; RFBM = x87|AVX skips SSE: MXCSR is still stored because RFBM[2] is set.
op_xrstor64_xsaveopt64_x87_avx PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov eax, 5
    xsaveopt64 [r8]
    ret
op_xrstor64_xsaveopt64_x87_avx ENDP

; MOVQ that only reads MM3 still clears TOP, marks every tag valid and leaves the x87 component in use.
op_xrstor64_movq_read_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    movq r9, mm3
    xsave64 [r8]
    emms
    ret
op_xrstor64_movq_read_xsave64 ENDP

; RFBM = AVX alone: XSAVE stores MXCSR and MXCSR_MASK with the YMM upper halves, x87 and XMM are not stored.
op_xrstor64_xsave64_avx PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov eax, 4
    xsave64 [r8]
    ret
op_xrstor64_xsave64_avx ENDP

; RFBM = AVX alone: MXCSR and MXCSR_MASK are stored, x87 and XMM are not.
op_xrstor64_xsaveopt64_avx PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov eax, 4
    xsaveopt64 [r8]
    ret
op_xrstor64_xsaveopt64_avx ENDP

; RFBM = AVX alone from [rcx]: standard format defines MXCSR, compacted format leaves caller MXCSR untouched;
; requires --entry-state canonical so the compacted path cannot observe entry MXCSR under the default profile.
op_xrstor64_avx_xsaveopt64_avx PROC FRAME
    .endprolog
    mov eax, 4
    xor edx, edx
    xrstor64 [rcx]
    xsaveopt64 [r8]
    ret
op_xrstor64_avx_xsaveopt64_avx ENDP

; SSE restore defines MXCSR before an AVX-only (compacted or standard) restore and XSAVEOPT64(4).
op_xrstor64_sse_compacted_avx_xsaveopt64_avx PROC FRAME
    .endprolog
    mov r10, rdx
    mov eax, 2
    xor edx, edx
    xrstor64 [rcx]
    mov eax, 4
    xrstor64 [r10]
    xsaveopt64 [r8]
    ret
op_xrstor64_sse_compacted_avx_xsaveopt64_avx ENDP

; The request mask comes from input_b (RAX = [rdx], RDX = [rdx+8]).
op_xsaveopt64_dynmask PROC FRAME
    .endprolog
    mov r9, rdx
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov rax, QWORD PTR [r9]
    mov rdx, QWORD PTR [r9+8]
    xsaveopt64 [r8]
    ret
op_xsaveopt64_dynmask ENDP

; --- XRSTOR / XRSTOR64 of the compacted format ---
; Every export first restores input_a (a standard image with every component live), so each component starts
; defined, then restores a compacted image and stores the whole state with xsave64 over the output.

; XSAVEC writes the compacted image into the output, XRSTOR of input_b (a standard image with other live state)
; replaces the registers, and XRSTOR of the SAME compacted buffer must bring input_a's state back. The header
; bytes [528..575] are zeroed first because xsavec leaves them alone and a compacted restore rejects them.
op_xsavec_xrstor_all PROC FRAME
    .endprolog
    mov r10, rdx
    xor r9d, r9d
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov QWORD PTR [r8+528], r9
    mov QWORD PTR [r8+536], r9
    mov QWORD PTR [r8+544], r9
    mov QWORD PTR [r8+552], r9
    mov QWORD PTR [r8+560], r9
    mov QWORD PTR [r8+568], r9
    xsavec64 [r8]
    xrstor64 [r10]
IFDEF ALTER_OP
    xrstor64 [r10]
ELSE
    xrstor64 [r8]
ENDIF
    xsave64 [r8]
    ret
op_xsavec_xrstor_all ENDP

; XSAVEC of x87|AVX leaves SSE out of XCOMP_BV: restoring RFBM = 7 initializes SSE and MXCSR (32-bit xrstor).
op_xsavec_xrstor_x87_avx PROC FRAME
    .endprolog
    mov r10, rdx
    xor r9d, r9d
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov QWORD PTR [r8+528], r9
    mov QWORD PTR [r8+536], r9
    mov QWORD PTR [r8+544], r9
    mov QWORD PTR [r8+552], r9
    mov QWORD PTR [r8+560], r9
    mov QWORD PTR [r8+568], r9
    mov eax, 5
    xsavec64 [r8]
    mov eax, 7
    xrstor64 [r10]
    xrstor [r8]
    xsave64 [r8]
    ret
op_xsavec_xrstor_x87_avx ENDP

; XSAVEC of x87|SSE leaves AVX out of XCOMP_BV: restoring RFBM = 7 initializes AVX.
op_xsavec_xrstor_x87_sse PROC FRAME
    .endprolog
    mov r10, rdx
    xor r9d, r9d
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov QWORD PTR [r8+528], r9
    mov QWORD PTR [r8+536], r9
    mov QWORD PTR [r8+544], r9
    mov QWORD PTR [r8+552], r9
    mov QWORD PTR [r8+560], r9
    mov QWORD PTR [r8+568], r9
    mov eax, 3
    xsavec64 [r8]
    mov eax, 7
    xrstor64 [r10]
    xrstor64 [r8]
    xsave64 [r8]
    ret
op_xsavec_xrstor_x87_sse ENDP

; RFBM = AVX alone restores only YMM[255:128]: x87, XMM and MXCSR keep input_b's values.
op_xsavec_xrstor_avx_only PROC FRAME
    .endprolog
    mov r10, rdx
    xor r9d, r9d
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov QWORD PTR [r8+528], r9
    mov QWORD PTR [r8+536], r9
    mov QWORD PTR [r8+544], r9
    mov QWORD PTR [r8+552], r9
    mov QWORD PTR [r8+560], r9
    mov QWORD PTR [r8+568], r9
    xsavec64 [r8]
    xrstor64 [r10]
    mov eax, 4
IFDEF ALTER_OP
    mov eax, 6
ENDIF
    xrstor64 [r8]
    mov eax, 7
    xsave64 [r8]
    ret
op_xsavec_xrstor_avx_only ENDP

; RFBM = SSE alone restores XMM and MXCSR from the compacted image; x87 and AVX keep input_b's values.
op_xsavec_xrstor_sse_only PROC FRAME
    .endprolog
    mov r10, rdx
    xor r9d, r9d
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov QWORD PTR [r8+528], r9
    mov QWORD PTR [r8+536], r9
    mov QWORD PTR [r8+544], r9
    mov QWORD PTR [r8+552], r9
    mov QWORD PTR [r8+560], r9
    mov QWORD PTR [r8+568], r9
    xsavec64 [r8]
    xrstor64 [r10]
    mov eax, 2
    xrstor64 [r8]
    mov eax, 7
    xsave64 [r8]
    ret
op_xsavec_xrstor_sse_only ENDP

; input_b is a hand-built compacted image and its RFBM comes from input_b itself (RAX = [rdx+832],
; RDX = [rdx+840]); the state it restores over input_a's is then stored with xsave64.
op_xrstor64_compacted_dynmask PROC FRAME
    .endprolog
    mov r10, rdx
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    mov rax, QWORD PTR [r10+832]
    mov rdx, QWORD PTR [r10+840]
    xrstor64 [r10]
    mov eax, 7
    xor edx, edx
    xsave64 [r8]
    ret
op_xrstor64_compacted_dynmask ENDP

; --- Chained AVX + SSE + MMX + XSAVE64 pipeline and dead-read fault ordering ---

op_pipeline PROC FRAME
    .endprolog
IFDEF ALTER_OP
    mov r9, QWORD PTR [rdx+8]
ELSE
    mov r9, QWORD PTR [rdx]
ENDIF
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    movq mm0, r9
    movq rax, mm0
    movq mm7, rax
    vmovdqu ymm0, YMMWORD PTR [rcx+832]
    vmovdqu ymm1, ymm0
    paddd xmm1, XMMWORD PTR [rcx+864]
    vmovdqu YMMWORD PTR [r8+832], ymm1
    mov eax, 7
    xsave64 [r8]
    emms
    ret
op_pipeline ENDP

; The discarded xrstor64 [r9] (from rdx) must fault after [r8] and before [r8+832].
op_discarded_read PROC FRAME
    .endprolog
    mov r9, rdx
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsave64 [r8]
    xrstor64 [r9]
    vmovdqu ymm0, YMMWORD PTR [rcx+576]
    vmovdqu YMMWORD PTR [r8+832], ymm0
    ret
op_discarded_read ENDP

END
