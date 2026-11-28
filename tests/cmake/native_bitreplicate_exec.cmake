if(X2W_HAVE_DXCAPI AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_bitreplicate_exec_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_bitreplicate_exec_check.py"
            controls "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_bitreplicate_exec_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_bitreplicate_exec_hardware
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_bitreplicate_exec_check.py"
            hardware "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_bitreplicate_exec_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
