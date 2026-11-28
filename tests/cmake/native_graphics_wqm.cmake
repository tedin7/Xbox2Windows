# Cross-quad WQM/EXECZ in a native pixel draw (#2907). native_graphics_wqm_hardware stays in CMakeLists.txt.
# The WARP entry runs every case that one quad per wave can decide; the IR entry checks, without a GPU, that the
# vote's quad and wave operations are emitted before any branch.
if(TARGET compile_dxil AND TARGET native_store_replay AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_graphics_wqm_warp COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_graphics_wqm_check.py" "$<TARGET_FILE:compile_dxil>"
        "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" warp)
    set_tests_properties(native_graphics_wqm_warp PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_test(NAME native_graphics_wqm_ir COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_graphics_wqm_check.py" ir "$<TARGET_FILE:native_pixel_ir_dump>")
endif()
