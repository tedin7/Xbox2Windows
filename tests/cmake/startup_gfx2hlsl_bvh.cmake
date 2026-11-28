# gfx2hlsl image_bvh_intersect_ray FP32 box-node slice: translator controls, a DXC compile when
# one is available, and a WARP/hardware dispatch when offscreen_check is built.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_bvh COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_bvh_check.py")
endif()
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_bvh_dispatch_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_bvh_check.py" dispatch "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_gfx2hlsl_bvh_dispatch_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
