# v_cvt_f16_u16, v_cvt_f16_i16, v_cvt_u16_f16 and v_cvt_i16_f16 in the direct C++ compute path.
# Named to sort after native_f16_alu.cmake so native_compute_ir_dump is already defined.
if(TARGET native_compute_ir_dump AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_cvt_f16_i16_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cvt_f16_i16_check.py" ir
        "$<TARGET_FILE:native_compute_ir_dump>")
    set_tests_properties(native_cvt_f16_i16_ir PROPERTIES TIMEOUT 60)
endif()
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_cvt_f16_i16_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cvt_f16_i16_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_f16_i16_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_cvt_f16_i16_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cvt_f16_i16_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_f16_i16_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
