# gfx2hlsl latch-first loop over the set bits of a wave OR: translation controls, then a triangle drawn with the
# translated pixel stage and compared with a CPU reference per desktop wave (mutant: highest bit first).
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_latch_loop_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_latch_loop_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_latch_loop_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_latch_loop_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_gfx2hlsl_latch_loop_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
