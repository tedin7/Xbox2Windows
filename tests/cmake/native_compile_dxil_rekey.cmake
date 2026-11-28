# Native cache re-keying (--rekey) in compile_dxil (#2935).
if(TARGET compile_dxil)
    add_test(NAME native_compile_dxil_rekey COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_compile_dxil_rekey_check.py"
        "$<TARGET_FILE:compile_dxil>")
    set_tests_properties(native_compile_dxil_rekey PROPERTIES TIMEOUT 60)
endif()
