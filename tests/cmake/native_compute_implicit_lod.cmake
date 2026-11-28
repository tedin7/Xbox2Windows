if(TARGET compile_dxil AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    set(implicit_lod_check "${CMAKE_SOURCE_DIR}/tests/native_compute_implicit_lod_check.py")
    add_test(NAME native_compute_implicit_lod_controls COMMAND "${INTAKE_PYTHON}" -B
        "${implicit_lod_check}" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_compute_implicit_lod_controls PROPERTIES TIMEOUT 330)
    if(TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME native_compute_implicit_lod_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${implicit_lod_check}" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}"
                "$<TARGET_FILE:offscreen_check>" ${adapter})
            set_tests_properties(native_compute_implicit_lod_${adapter} PROPERTIES TIMEOUT 540 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
