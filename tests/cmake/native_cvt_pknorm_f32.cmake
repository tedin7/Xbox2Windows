# Direct DXIL VOP3 v_cvt_pknorm_i16_f32 and v_cvt_pknorm_u16_f32. Compute replay needs SM 6.6 with wave32.
if(TARGET compile_dxil AND TARGET native_store_replay AND TARGET offscreen_check AND
   EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(checker "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_cvt_pknorm_f32_check.py")
    add_test(NAME native_cvt_pknorm_f32_ir COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" ir "$<TARGET_FILE:native_compute_ir_dump>")
    set_tests_properties(native_cvt_pknorm_f32_ir PROPERTIES TIMEOUT 60)
    add_test(NAME native_cvt_pknorm_f32_controls COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_pknorm_f32_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_cvt_pknorm_f32_compute_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" compute "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_cvt_pknorm_f32_compute_hardware PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_cvt_pknorm_f32_pixel_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${checker}" pixel "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(native_cvt_pknorm_f32_pixel_${adapter} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
    unset(checker)
endif()
