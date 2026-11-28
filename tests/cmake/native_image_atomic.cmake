if(TARGET compile_dxil AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(checker "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_image_atomic_check.py")
    add_test(NAME native_image_atomic_controls COMMAND "${INTAKE_PYTHON}" -B
        "${checker}" controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_image_atomic_controls PROPERTIES TIMEOUT 60)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_image_atomic_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${checker}" hardware "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(native_image_atomic_${adapter} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endforeach()
    if(TARGET startup_stream_consumer)
        add_test(NAME native_image_atomic_runtime_hardware COMMAND "${INTAKE_PYTHON}" -B
            "${checker}" runtime "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}"
            "$<TARGET_FILE:offscreen_check>" "$<TARGET_FILE:startup_stream_consumer>" hardware)
        set_tests_properties(native_image_atomic_runtime_hardware PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endif()
    unset(checker)
endif()
