# gfx2hlsl v_ldexp_f32 lowering (Refs #2618):
# validates that v_ldexp_f32 lowers to ldexp32 and refuses invalid modifiers.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_ldexp_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_ldexp_check.py")
endif()
