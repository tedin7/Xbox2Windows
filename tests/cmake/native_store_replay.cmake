if(TARGET native_store_replay AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_store_replay_adapter_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_store_replay_adapter_check.py"
        "$<TARGET_FILE:native_store_replay>")
    set_tests_properties(native_store_replay_adapter_controls PROPERTIES TIMEOUT 30)
endif()
