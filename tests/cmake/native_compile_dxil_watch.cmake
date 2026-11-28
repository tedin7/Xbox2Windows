# Continuous and one-shot --watch mode for compile_dxil (#2935).
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_compile_dxil_watch COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_compile_dxil_watch_check.py"
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_compile_dxil_watch PROPERTIES TIMEOUT 90)
endif()
