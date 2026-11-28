# Direct compute VOP3P 4-bit nibble dots. Replay needs SM 6.6 with a physical wave32 adapter.
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_packed_dot8_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_packed_dot8_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_packed_dot8_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_packed_dot8_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_packed_dot8_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_packed_dot8_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
