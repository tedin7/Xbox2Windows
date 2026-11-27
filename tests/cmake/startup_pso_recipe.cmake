# Device-free checks for the opt-in translated-PSO recipe format
# (src/startup_stubs/pso_recipe.h).
add_executable(startup_pso_recipe_check tests/startup_pso_recipe_check.cpp)
target_include_directories(startup_pso_recipe_check PRIVATE src)
target_compile_features(startup_pso_recipe_check PRIVATE cxx_std_17)
add_test(NAME startup_pso_recipe COMMAND startup_pso_recipe_check)
