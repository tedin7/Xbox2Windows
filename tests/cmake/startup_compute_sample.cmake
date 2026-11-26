if(EXISTS "${INTAKE_PYTHON}")
    set(compute_sample_check "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_dispatch_texture_check.py")
    add_test(NAME startup_compute_sample_controls COMMAND "${INTAKE_PYTHON}" -B
        "${compute_sample_check}" sampler-controls)
    if(DXC_EXECUTABLE AND TARGET startup_stream_consumer)
        foreach(sample_case IN ITEMS table implicit)
            add_test(NAME startup_compute_sample_${sample_case}_warp COMMAND "${INTAKE_PYTHON}" -B
                "${compute_sample_check}" "$<TARGET_FILE:startup_stream_consumer>"
                "${DXC_EXECUTABLE}" warp ${sample_case})
            set_tests_properties(startup_compute_sample_${sample_case}_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
    unset(compute_sample_check)
endif()
