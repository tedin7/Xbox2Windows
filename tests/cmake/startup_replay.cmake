# x2w_replay: a recording of startup_replay_quad replays to the same pictures, and a recording with one clear colour
# changed replays to a different last picture only (tests/startup_replay_check.py).
if(TARGET startup_replay_quad AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()

# x2w_replay writes a recording's pages (PAGE/MAP) at their recorded addresses, in sequence order; a range that cannot be
# had stops it with exit 3 (tests/startup_replay_pages_play_check.py).
if(TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_replay_pages_play COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_pages_play_check.py"
        "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_pages_play PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

# Replay of the texture / sampler / image family (row c17 of briefs/replay_all_hands.md): each scene is an in-repo stand-in check
# that takes a backend and runs under the settings of its own test (copied from it). It is recorded (X2W_REPLAY_RECORD=all),
# replayed by x2w_replay, and every call must play on a known object, with equal pictures where the scene presents
# (tests/startup_replay_family_check.py). Deferred: the scene tests are registered elsewhere in the configure.
function(x2w_replay_family_scene scene own_test)
    foreach(backend IN ITEMS warp hardware)
        string(REPLACE "@B@" "${backend}" own "${own_test}")
        if(NOT TARGET ${scene} OR NOT TEST ${own})
            continue()
        endif()
        get_test_property(${own} ENVIRONMENT scene_env)
        if(scene_env STREQUAL "scene_env-NOTFOUND")
            set(scene_env "")
        endif()
        add_test(NAME startup_replay_family_${scene}_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_family_check.py" "$<TARGET_FILE:x2w_replay>" ${backend} --
            "$<TARGET_FILE:${scene}>" ${backend})
        set_tests_properties(startup_replay_family_${scene}_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77
            ENVIRONMENT "${scene_env}")
    endforeach()
endfunction()
function(x2w_replay_family_all)
    x2w_replay_family_scene(startup_view_dimension_device startup_view_dimension_@B@)
    # Copies and uploads (row c13): list copies, upload arena, arena lifetime, page reuse.
    x2w_replay_family_scene(startup_list_copy startup_list_copy_@B@)
    x2w_replay_family_scene(startup_upload_arena startup_upload_arena_@B@)
    x2w_replay_family_scene(startup_upload_lifetime startup_upload_lifetime_@B@)
    x2w_replay_family_scene(startup_upload_page_reuse startup_upload_page_reuse_@B@)
    x2w_replay_family_scene(startup_list_copy_texture_warp startup_list_copy_texture_@B@)
    x2w_replay_family_scene(startup_resource_retire_views startup_resource_retire_views_@B@_debug0)
    # startup_clear_uav is not a family scene: its textures come from the desktop device, outside the recording.
    x2w_replay_family_scene(startup_list_dispatch_native startup_list_dispatch_native_@B@)
    x2w_replay_family_scene(startup_small_texture_4k_device_check startup_small_texture_4k_device_@B@)
    x2w_replay_family_scene(startup_slot133_lazy_textures_check startup_slot133_lazy_textures_@B@_debug0)
    # startup_null_views_check stays out until the player maps a descriptor heap backed by title memory (open gap).
endfunction()
if(TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL x2w_replay_family_all)
endif()
