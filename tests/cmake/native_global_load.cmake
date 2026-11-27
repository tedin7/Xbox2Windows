# Direct C++ compute: GLOBAL DWORD loads through a root SRV address. Refusals and the module's guards run
# everywhere; the dispatch readback needs shader model 6.6 with wave32 and otherwise reports a skip (77).
if(EXISTS "${INTAKE_PYTHON}" AND TARGET compile_dxil AND TARGET native_store_replay)
    add_test(NAME native_global_load_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_global_load_check.py" "$<TARGET_FILE:compile_dxil>"
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_global_load_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
