if(EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer AND TARGET offscreen_check)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_waterfall_heap_oracle_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_waterfall_heap_check.py"
            --oracle "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_compute_waterfall_heap_oracle_${adapter} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
        add_test(NAME startup_compute_waterfall_heap_dispatch_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_waterfall_heap_check.py"
            --dispatch "$<TARGET_FILE:startup_stream_consumer>" "${DXC_EXECUTABLE}" ${adapter})
        set_tests_properties(startup_compute_waterfall_heap_dispatch_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
