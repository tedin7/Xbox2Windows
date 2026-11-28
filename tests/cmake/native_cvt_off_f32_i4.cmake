# Direct compute VOP1/VOP3A v_cvt_off_f32_i4. Replay needs SM 6.6 with a physical wave32 adapter.
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_cvt_off_f32_i4_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_cvt_off_f32_i4_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_off_f32_i4_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_cvt_off_f32_i4_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_cvt_off_f32_i4_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_off_f32_i4_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
