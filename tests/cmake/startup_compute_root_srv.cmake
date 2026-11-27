if(EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_root_srv_dispatch_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_root_srv_dispatch_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${DXC_EXECUTABLE}" ${adapter})
        set_tests_properties(startup_compute_root_srv_dispatch_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
