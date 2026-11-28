# X2W_INPUT_TRACE=1 logs the resolved gamepad state and the script's marks and waits. The test runs
# the same script with the switch on and off; the control must not produce an x2w-input-trace line.
add_executable(startup_gameinput_trace tests/startup_gameinput_trace.cpp)
target_include_directories(startup_gameinput_trace PRIVATE src)
target_compile_features(startup_gameinput_trace PRIVATE cxx_std_17)
add_dependencies(startup_gameinput_trace startup_stub_gameinput)
add_test(NAME startup_gameinput_trace COMMAND startup_gameinput_trace "$<TARGET_FILE:startup_stub_gameinput>")
set_tests_properties(startup_gameinput_trace PROPERTIES RUN_SERIAL TRUE TIMEOUT 60 SKIP_RETURN_CODE 77)
