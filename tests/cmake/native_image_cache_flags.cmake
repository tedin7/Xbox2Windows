if(TARGET compile_dxil AND TARGET offscreen_check AND EXISTS "${INTAKE_PYTHON}" AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_image_cache_flags_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_image_cache_flags_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_image_cache_flags_controls PROPERTIES TIMEOUT 60)
    foreach(stage IN ITEMS ps cs)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME native_image_cache_flags_${stage}_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${CMAKE_SOURCE_DIR}/tests/native_image_cache_flags_check.py" ${stage}
                "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" "$<TARGET_FILE:offscreen_check>" ${adapter})
            set_tests_properties(native_image_cache_flags_${stage}_${adapter} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
        endforeach()
    endforeach()
endif()
