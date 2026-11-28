if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_s_not_b64_exec_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_s_not_b64_exec_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_s_not_b64_exec_controls PROPERTIES TIMEOUT 60)
    if(TARGET native_store_replay)
        add_test(NAME native_s_not_b64_exec_hardware COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_s_not_b64_exec_check.py" hardware
            "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
        set_tests_properties(native_s_not_b64_exec_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endif()
endif()
