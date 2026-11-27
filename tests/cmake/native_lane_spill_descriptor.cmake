if(TARGET compile_dxil AND TARGET native_store_replay AND
   EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_lane_spill_descriptor_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_lane_spill_descriptor_check.py" "$<TARGET_FILE:compile_dxil>"
            "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}" "${adapter}")
        set_tests_properties(native_lane_spill_descriptor_${adapter} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
endif()
