if(TARGET compile_dxil AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_sample_ssamp_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_sample_ssamp_check.py"
            "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(native_sample_ssamp_${adapter} PROPERTIES TIMEOUT 150 SKIP_RETURN_CODE 77)
    endforeach()
endif()
