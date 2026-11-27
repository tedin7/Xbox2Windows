# Direct compute SOP2 s_lshl{1,2,3,4}_add_u32. Replay needs SM 6.6 with a physical wave32 adapter.
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_s_lshl_add_u32_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_s_lshl_add_u32_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_s_lshl_add_u32_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_s_lshl_add_u32_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_s_lshl_add_u32_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_s_lshl_add_u32_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
