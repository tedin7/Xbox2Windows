# Pass thermometer (X2W_PASS_STATS=1, docs/startup-runtime.md#pass-thermometer): a clear, a translated exposure
# dispatch and a copy to the presented surface, each reduced on the GPU and compared with CPU statistics of the same
# texels read back; the collapse line must name the exposure pass, and the switch off must write nothing.
if(TARGET startup_stream_consumer AND EXISTS "${INTAKE_PYTHON}")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_pass_stats_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_pass_stats_check.py" "$<TARGET_FILE:startup_stream_consumer>"
            "${DXC_EXECUTABLE}" ${adapter})
        set_tests_properties(startup_pass_stats_${adapter} PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77)
    endforeach()
endif()
