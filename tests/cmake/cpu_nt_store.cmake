# Legacy non-temporal stores in the shared lowering (MOVNTPS, MOVNTPD, MOVNTDQ, MOVNTI): each lowers to its
# streaming-store intrinsic, the XMM forms keep the 16-byte alignment fault, and the MMX, SSE4a and VEX
# neighbours stay refused. The fixture compares native bytes with the lowered C++ for stores, misaligned
# targets and a PAGE_NOACCESS target, with controls that remove the alignment test or the store.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME cpu_nt_store COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu/nt_store_check.py")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/nt_store_fixture.cpp"
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/cpu/nt_store_fixture.py" emit
            "${CMAKE_CURRENT_BINARY_DIR}/nt_store_fixture.cpp"
        DEPENDS tests/cpu/nt_store_fixture.py tools/ffx_version_query.py VERBATIM)
    add_executable(cpu_nt_store_fixture "${CMAKE_CURRENT_BINARY_DIR}/nt_store_fixture.cpp")
    target_compile_features(cpu_nt_store_fixture PRIVATE cxx_std_17)
    add_test(NAME cpu_nt_store_fixture COMMAND cpu_nt_store_fixture)
    set_tests_properties(cpu_nt_store_fixture PROPERTIES SKIP_RETURN_CODE 77)
endif()
