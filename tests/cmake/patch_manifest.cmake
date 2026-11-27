# Patch manifest schema, loader and hash gate (tools/patch_manifest.py).
if(EXISTS "${INTAKE_PYTHON}")
    add_test(
        NAME patch_manifest_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/patch_manifest_check.py"
    )
endif()
