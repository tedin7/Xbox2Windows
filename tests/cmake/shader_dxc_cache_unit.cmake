# Unit and mutation suite for tools/dxc_cache.py (atomic restore, symlink/file-backed
# cache_dir fallback, -Fo dir/symlink refusal, multi-output -Fc/-Fd, corrupt cache miss,
# and -I/#include bypass).
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME shader_dxc_cache_unit_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/dxc_cache_unit_check.py")
    set_tests_properties(shader_dxc_cache_unit_check PROPERTIES TIMEOUT 60)
endif()
