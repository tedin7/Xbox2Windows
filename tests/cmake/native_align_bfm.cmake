# Direct C++ compute: v_alignbit_b32, v_alignbyte_b32 and v_bfm_b32 (#2985). The controls compile and check the
# reference's mutants; the readback needs shader model 6.6 with wave32 and otherwise reports a skip (77).
if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_align_bfm_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_align_bfm_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_align_bfm_controls PROPERTIES TIMEOUT 60)
    add_test(NAME native_align_bfm_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_align_bfm_check.py" hardware
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_align_bfm_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()
