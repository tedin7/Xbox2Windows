# The mesh replay oracle (X2W_REPLAY_ORACLE=mesh): proof and serial wave-loop translations of one slot 57 key drawn
# by the stand-in from the same inputs, compared; a wrong proof translation must be reported as differs.
if(EXISTS "${INTAKE_PYTHON}")
    foreach(adapter IN ITEMS hardware warp)
        add_test(NAME startup_mesh57_oracle_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_oracle_check.py" "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} "$<TARGET_FILE:startup_indirect_draw_indexed>")
        set_tests_properties(startup_mesh57_oracle_${adapter} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    endforeach()
endif()
