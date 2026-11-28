# A slot 67 resource-heap descriptor address formed by s_add_u32 with its carry (#3224).
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_heap_offset_add COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_heap_offset_add_check.py"
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_heap_offset_add PROPERTIES TIMEOUT 60)
endif()
