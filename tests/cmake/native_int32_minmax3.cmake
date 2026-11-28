# Direct C++ compute 32-bit integer min3/max3/med3 regression.
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_int32_minmax3_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_int32_minmax3_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_int32_minmax3_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_int32_minmax3_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_int32_minmax3_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_int32_minmax3_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
