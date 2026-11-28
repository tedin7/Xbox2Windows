# --prefix-live: which prefix calls the replay can leave out (src/replay/prefix_live.h). Pure logic, no D3D12.
add_executable(startup_replay_prefix_live_check tests/startup_replay_prefix_live_check.cpp)
target_include_directories(startup_replay_prefix_live_check PRIVATE src)
target_compile_features(startup_replay_prefix_live_check PRIVATE cxx_std_17)
add_test(NAME startup_replay_prefix_live COMMAND startup_replay_prefix_live_check)

# x2w_replay: a recording of startup_replay_quad replays to the same pictures, and a recording with one clear colour
# changed replays to a different last picture only (tests/startup_replay_check.py).
if(TARGET startup_replay_quad AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        add_test(NAME startup_replay_slot_layouts_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" slot-layouts)
        add_test(NAME startup_replay_null_views_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" null-views)
        add_test(NAME startup_replay_paced_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" paced)
        add_test(NAME startup_replay_execute_slot35_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" execute-slot35)
        add_test(NAME startup_replay_static_list_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" static-list)
        add_test(NAME startup_replay_static_release_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" static-release)
        add_test(NAME startup_replay_infer_execute_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}" infer-execute)
        set_tests_properties(startup_replay_${backend} startup_replay_slot_layouts_${backend} startup_replay_null_views_${backend}
            startup_replay_paced_${backend} startup_replay_execute_slot35_${backend} startup_replay_static_list_${backend} startup_replay_static_release_${backend} startup_replay_infer_execute_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME startup_replay_title_ranges_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_title_ranges_check.py"
        "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_title_ranges_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    add_test(NAME startup_replay_draw_pages_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_draw_pages_check.py"
        "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_draw_pages_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

    add_test(NAME startup_replay_gpu_state_warp
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_gpu_state_check.py"
        "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_gpu_state_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    add_test(NAME startup_replay_draw_tables_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_draw_tables_check.py"
        "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_draw_tables_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
endif()

# x2w_replay writes a recording's pages (PAGE/MAP) at their recorded addresses, in sequence order; a range that cannot be
# had stops it with exit 3 (tests/startup_replay_pages_play_check.py).
if(TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_replay_pages_play COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_pages_play_check.py"
        "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_pages_play PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

add_executable(startup_replay_submit_pages tests/startup_replay_submit_pages.cpp)
target_compile_features(startup_replay_submit_pages PRIVATE cxx_std_17)
target_link_libraries(startup_replay_submit_pages PRIVATE d3d12_xs_shim d3d12 dxgi)
if(EXISTS "${INTAKE_PYTHON}")
    foreach(backend warp hardware)
        add_test(NAME startup_replay_submit_pages_${backend}
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_submit_pages_check.py"
                "$<TARGET_FILE:startup_replay_submit_pages>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_submit_pages_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()

if(TARGET startup_resource_budget AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_placedheap_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_placedheap_check.py"
            "$<TARGET_FILE:startup_resource_budget>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_placedheap_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()

# A synthetic compute dispatch writes to a placed UAV that aliases title memory.
if(TARGET startup_stream_consumer AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}" AND DXC_EXECUTABLE)
    foreach(backend warp hardware)
        add_test(NAME startup_replay_compute_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_compute_check.py" "$<TARGET_FILE:startup_stream_consumer>"
            "$<TARGET_FILE:x2w_replay>" "${DXC_EXECUTABLE}" "${backend}")
        set_tests_properties(startup_replay_compute_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()

# Submission dumps compare scenes without Present, including a changed-clear control.
if(TARGET startup_replay_quad AND TARGET startup_stream_consumer AND TARGET x2w_replay
    AND DXC_EXECUTABLE AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_dump_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_dump_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:startup_stream_consumer>" "$<TARGET_FILE:x2w_replay>"
            "${DXC_EXECUTABLE}" "${backend}")
        set_tests_properties(startup_replay_dump_${backend} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    endforeach()
endif()

if(TARGET x2w_replay)
    add_executable(startup_replay_invoke tests/startup_replay_invoke.cpp src/replay/replay_invoke.asm)
    target_compile_features(startup_replay_invoke PRIVATE cxx_std_17)
    add_test(NAME startup_replay_invoke COMMAND startup_replay_invoke)
    if(EXISTS "${INTAKE_PYTHON}")
        add_test(NAME startup_replay_call_formats COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_call_formats_check.py" "$<TARGET_FILE:x2w_replay>")
        set_tests_properties(startup_replay_call_formats PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
endif()
