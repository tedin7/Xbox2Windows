# Direct C++ image path: image_gather4_c_lz as textureGatherCmp (#2969). Module and refusal checks, then a graphics
# pipeline created on WARP through native_recording_check --graphics-sampler.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET compile_dxil AND TARGET native_recording_check AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_gather_compare_warp COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_gather_compare_check.py" "$<TARGET_FILE:compile_dxil>"
        "${DXC_BIN_DIR}" "$<TARGET_FILE:native_recording_check>")
    set_tests_properties(native_gather_compare_warp PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
