# Direct compute VOP2 signed byte dot-accumulate.
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_dot4c_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_dot4c_check.py" ir
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_dot4c_ir PROPERTIES TIMEOUT 60)
    add_test(NAME native_dot4c_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_dot4c_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_dot4c_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
