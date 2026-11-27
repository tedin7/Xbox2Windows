# Graphics root SRV: no desktop copy that no draw reads (opt-in X2W_D3D12_ROOT_SRV_AT_DRAW=1).
# A translated draw reads a slot 44 root SRV whose title bytes change before the draw; it runs with the switch unset
# (slot 44 copies) and set (no copy).
target_sources(startup_stream_consumer PRIVATE tests/startup_root_srv_draw.cpp)
if(EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_root_srv_draw_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_root_srv_draw_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" ${backend})
        set_tests_properties(startup_root_srv_draw_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
