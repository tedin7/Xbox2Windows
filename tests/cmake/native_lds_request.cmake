if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_lds_request COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_lds_request_check.py" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
endif()
