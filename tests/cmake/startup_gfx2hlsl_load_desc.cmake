# gfx2hlsl compute buffer load through a modified descriptor (#2901):
# validates that compute buffer loads through a modified or overwritten descriptor word are refused.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_load_desc_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_load_desc_check.py")
endif()
