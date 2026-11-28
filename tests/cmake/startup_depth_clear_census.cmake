# depth_clear_census.h: the log-only census of depth clears and of decoded GREATER depth tests by the view's last clear.
add_executable(startup_depth_clear_census_check tests/startup_depth_clear_census_check.cpp)
target_compile_features(startup_depth_clear_census_check PRIVATE cxx_std_17)
target_include_directories(startup_depth_clear_census_check PRIVATE src)
add_test(NAME startup_depth_clear_census COMMAND startup_depth_clear_census_check)
