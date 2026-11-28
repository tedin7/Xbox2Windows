# Direct DXIL !llvm.ident metadata check and WARP/hardware draw verification (#2915).
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_dxil_llvm_ident_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_dxil_llvm_ident_check.py" container
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_dxil_llvm_ident_controls PROPERTIES TIMEOUT 60)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_dxil_llvm_ident_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_dxil_llvm_ident_check.py" draw
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_dxil_llvm_ident_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
endif()
