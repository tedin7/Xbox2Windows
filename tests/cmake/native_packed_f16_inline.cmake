if(TARGET native_compute_ir_dump AND TARGET compile_dxil AND TARGET native_store_replay AND TARGET offscreen_check AND
   EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(checker "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_packed_f16_inline_check.py")
    add_test(NAME native_packed_f16_inline_ir COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" ir "$<TARGET_FILE:native_compute_ir_dump>")
    set_tests_properties(native_packed_f16_inline_ir PROPERTIES TIMEOUT 60)
    add_test(NAME native_packed_f16_inline_controls COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_packed_f16_inline_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_packed_f16_inline_compute_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" compute "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_packed_f16_inline_compute_hardware PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_packed_f16_inline_pixel_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${checker}" pixel "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(native_packed_f16_inline_pixel_${adapter} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
    unset(checker)
endif()
