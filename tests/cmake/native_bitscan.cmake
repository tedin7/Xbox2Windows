# Direct C++ compute: 32-bit bit counts and scans (s_bcnt0/1, s_ff0/1, s_flbit, v_ffbl, v_ffbh). The controls
# compile and check the reference's mutants; the readback needs shader model 6.6 with wave32 and otherwise reports
# a skip (77).
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_bitscan_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_bitscan_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_bitscan_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_bitscan_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_bitscan_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_bitscan_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
