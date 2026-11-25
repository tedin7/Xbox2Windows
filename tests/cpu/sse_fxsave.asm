; coverage: source cpu_aot_sse_fxsave
; Source-built MASM fixture exercising 4 SSE state forms from docs/coverage/x64_zen2.tsv
; group SSE: fxsave, fxsave64, fxrstor, and fxrstor64, along with transport/ABI glue
; (movdqa, paddd, movq, emms).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (16-byte aligned; 512+ bytes)
;   RDX = const void* input_b (16-byte aligned; 512+ bytes)
;   R8  = void*       output  (16-byte aligned; 512+ bytes)
.code

; --- Entry-state FXSAVE / FXSAVE64 (opt-in canonical-entry profile) ---
; Native callers targeting lifted canonical exports must establish canonical FX state
; (FCW=0x037F, FSW=0, FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0, MXCSR=0x1F80, zero ST/XMM).

op_fxsave_only PROC FRAME
    .endprolog
    fxsave [r8]
    ret
op_fxsave_only ENDP

op_fxsave64_only PROC FRAME
    .endprolog
    fxsave64 [r8]
    ret
op_fxsave64_only ENDP

; --- FXRSTOR / FXRSTOR64 -> FXSAVE / FXSAVE64 roundtrips ---

op_fxrstor_fxsave PROC FRAME
    .endprolog
    fxrstor [rcx]
    fxsave [r8]
    ret
op_fxrstor_fxsave ENDP

op_fxrstor64_fxsave64 PROC FRAME
    .endprolog
    fxrstor64 [rcx]
IFDEF ALTER_OP
    fxsave [r8]
ELSE
    fxsave64 [r8]
ENDIF
    ret
op_fxrstor64_fxsave64 ENDP

op_fxrstor_fxsave64 PROC FRAME
    .endprolog
IFDEF ALTER_OP
    fxrstor64 [rcx]
ELSE
    fxrstor [rcx]
ENDIF
    fxsave64 [r8]
    ret
op_fxrstor_fxsave64 ENDP

op_fxrstor64_fxsave PROC FRAME
    .endprolog
    fxrstor64 [rcx]
    fxsave [r8]
    ret
op_fxrstor64_fxsave ENDP

op_fxrstor64_disp PROC FRAME
    .endprolog
IFDEF ALTER_OP
    fxrstor [rcx+64]
ELSE
    fxrstor64 [rcx+64]
ENDIF
    fxsave64 [r8+64]
    ret
op_fxrstor64_disp ENDP

; --- Chained SSE + MMX + FXSAVE64 pipeline and dead-read fault ordering ---

op_pipeline PROC FRAME
    .endprolog
    fxrstor64 [rcx]
    movq mm0, QWORD PTR [rdx]
    movq rax, mm0
    movq mm7, rax
    movdqa xmm0, XMMWORD PTR [rdx+16]
    paddd xmm0, XMMWORD PTR [rdx+32]
    movdqa xmm1, xmm0
    paddd xmm1, xmm2
    movdqa XMMWORD PTR [r8+512], xmm1
IFDEF ALTER_OP
    fxsave [r8]
ELSE
    fxsave64 [r8]
ENDIF
    emms
    ret
op_pipeline ENDP

; MOVQ that only reads MM3, to memory or to a register, still clears TOP and marks every tag valid.
op_fxrstor64_movq_store PROC FRAME
    .endprolog
    fxrstor64 [rcx]
    movq QWORD PTR [r8+512], mm3
    fxsave64 [r8]
    emms
    ret
op_fxrstor64_movq_store ENDP

op_fxrstor64_movq_gpr PROC FRAME
    .endprolog
    fxrstor64 [rcx]
    movq r9, mm3
    fxsave64 [r8]
    emms
    ret
op_fxrstor64_movq_gpr ENDP

; The discarded fxrstor64 [rdx] must fault after [r8] and before [r8+512].
op_discarded_read PROC FRAME
    .endprolog
    fxrstor64 [rcx]
    fxsave64 [r8]
    fxrstor64 [rdx]
    movdqa xmm0, XMMWORD PTR [rcx+160]
    movdqa XMMWORD PTR [r8+512], xmm0
    ret
op_discarded_read ENDP

END
