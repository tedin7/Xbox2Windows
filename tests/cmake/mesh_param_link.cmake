if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME mesh_param_link_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/mesh_param_link_check.py")
    foreach(adapter IN ITEMS hardware warp)
        # Mesh param0..param31 exports paired with gfx2hlsl MeshPixelStage (#2413 T4).
        add_test(NAME mesh_param_link_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/mesh_param_link_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(mesh_param_link_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endforeach()
endif()
