if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_mesh57_waveloop_walls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_waveloop_walls_check.py")
endif()
