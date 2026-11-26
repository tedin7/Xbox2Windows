# Replay family: resource creation, placed resources, heaps. Each scene below is an existing fixture, run unchanged; it is
# recorded (X2W_REPLAY_RECORD=all), replayed by x2w_replay and must log the same creations (tests/startup_replay_family_check.py).
# The CONTROL variant damages one byte of the first creation's description memory in the recording: the replay must differ.
if(TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    set(replay_family_env "X2W_D3D12_BUDGET=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_D3D12_RENDER=1;X2W_D3D12_DEBUG=0")
    function(startup_replay_family name backend env)
        set(scene_args ${ARGN})
        add_test(NAME startup_replay_family_${name}_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_family_check.py" "$<TARGET_FILE:x2w_replay>" "${backend}" --
            "$<TARGET_FILE:${name}>" ${backend} ${scene_args})
        set_tests_properties(startup_replay_family_${name}_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77
            ENVIRONMENT "${env}")
    endfunction()
    foreach(backend IN ITEMS warp hardware)
        if(TARGET startup_resource_retire)
            startup_replay_family(startup_resource_retire ${backend} "${replay_family_env};X2W_D3D12_RELEASE_RETIRED=1")
        endif()
        if(TARGET startup_resource_retire_title)
            startup_replay_family(startup_resource_retire_title ${backend} "${replay_family_env};X2W_RESOURCE_INDEX_CHECK=1;X2W_D3D12_RELEASE_RETIRED=1")
        endif()
        if(TARGET startup_resource_budget)
            startup_replay_family(startup_resource_budget ${backend} "${replay_family_env}")
        endif()
        if(TARGET startup_resource_desc)
            startup_replay_family(startup_resource_desc ${backend} "${replay_family_env}")
        endif()
        if(TARGET startup_committed_heap_state_check)
            startup_replay_family(startup_committed_heap_state_check ${backend} "")
        endif()
        if(TARGET startup_committed_slot76_check)
            startup_replay_family(startup_committed_slot76_check ${backend} "")
        endif()
    endforeach()
    # Negative control on one scene: a damaged creation must not replay to the same creations.
    if(TARGET startup_resource_desc)
        add_test(NAME startup_replay_family_control COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_family_check.py" "$<TARGET_FILE:x2w_replay>" warp --control --
            "$<TARGET_FILE:startup_resource_desc>" warp)
        set_tests_properties(startup_replay_family_control PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "${replay_family_env}")
    endif()
endif()
