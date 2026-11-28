if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_dpp8_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_dpp8_check.py"
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_dpp8_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
