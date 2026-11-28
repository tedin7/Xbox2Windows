# XMem page-run fixtures (#2911): readers inside a run break, the cross-call merge of shared views, the slot 31
# placeholder split, split-before-map inside a placeholder, and a DirectStorage destination inside a run break.
# Readers that do not fault, inside a run break held open by the test-controls build: x2w_xmem_stable_call,
# XMemVirtualQuery, the VirtualQueryEx import wrapper and the stand-in readers that go through
# x2w_local::outside_run_breaks must answer after the break. Control: a copy of local_log.h that asks xmem.dll for
# a missing export, so the stand-in readers run directly and answer inside the window.
foreach(variant check control)
    add_executable(xmem_run_break_readers_${variant} tests/xmem_run_break_readers_check.cpp)
    target_compile_features(xmem_run_break_readers_${variant} PRIVATE cxx_std_17)
    add_dependencies(xmem_run_break_readers_${variant} xmem_shim_test_controls)
endforeach()
target_include_directories(xmem_run_break_readers_check PRIVATE src)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/startup_stubs/local_log.h" local_log_text)
string(REPLACE "\"x2w_xmem_stable_call\"" "\"x2w_xmem_stable_call_control\"" local_log_control_text "${local_log_text}")
if(local_log_control_text STREQUAL local_log_text)
    message(FATAL_ERROR "local_log.h lost the anchor used by xmem_run_break_readers_control")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/run_break_readers_control/startup_stubs/local_log.h" "${local_log_control_text}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/startup_stubs/local_log.h)
target_include_directories(xmem_run_break_readers_control BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/run_break_readers_control")
target_include_directories(xmem_run_break_readers_control PRIVATE src)
add_test(NAME xmem_run_break_readers COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>")
add_test(NAME xmem_run_break_readers_control COMMAND xmem_run_break_readers_control
         "$<TARGET_FILE:xmem_shim_test_controls>" control)
add_test(NAME xmem_run_break_readers_next_break_control COMMAND xmem_run_break_readers_check
         "$<TARGET_FILE:xmem_shim_test_controls>" next-break-control)
# Default mode (no X2W_XMEM_SHARED_VIEWS): readers that overlap an unmap answer inside it, as before. Lazy commit:
# the readers' output buffer in a mapped slot not yet committed.
add_test(NAME xmem_run_break_readers_default COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>" default)
add_test(NAME xmem_run_break_readers_lazy COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>" lazy)
# A fault handler that looks the page up after a later break on another run has begun: a second unmap on a disjoint
# run, and one permissive call that joins the CPU run and then its GPU mirror run (X2W_XMEM_GPU_MIRROR; the program
# sets X2W_XMEM_JOIN_CALLS=1 for `mirror` only).
# stamps: a held fault whose break is followed by 63, 64 or 200 later breaks on another page is retried each time
# (the run keeps the stamp of the break that mapped it again); a fault that stays is passed on at once on a run no
# break mapped, and after one retry on a run a break mapped.
add_test(NAME xmem_run_break_readers_two_runs COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>" two-runs)
add_test(NAME xmem_run_break_readers_mirror COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>" mirror)
add_test(NAME xmem_run_break_readers_stamps COMMAND xmem_run_break_readers_check "$<TARGET_FILE:xmem_shim_test_controls>" stamps)
set_tests_properties(xmem_run_break_readers xmem_run_break_readers_control xmem_run_break_readers_next_break_control
                     xmem_run_break_readers_default xmem_run_break_readers_lazy xmem_run_break_readers_two_runs
                     xmem_run_break_readers_mirror xmem_run_break_readers_stamps PROPERTIES TIMEOUT 60)
set_tests_properties(xmem_run_break_readers_default PROPERTIES
    PASS_REGULAR_EXPRESSION "default mode: readers answer inside an unmap" FAIL_REGULAR_EXPRESSION "check_failed")
# Cross-call merge of shared views (merge_with_previous, X2W_XMEM_JOIN_CALLS=1 set by the program): single-page calls
# on consecutive slots join one view, and the steps around it (an unmap inside, slots out of order, a reader inside the
# held window, refused steps, two reservations). The control keeps one view per call (X2W_XMEM_TEST_NO_MERGE=1, test
# build only); the deployed shim ignores that switch; without X2W_XMEM_SHARED_VIEWS every page keeps a view of its
# own. no_join: the deployed shim with shared views and without X2W_XMEM_JOIN_CALLS keeps a view per call.
add_executable(xmem_cross_call_merge_check tests/xmem_cross_call_merge_check.cpp)
target_compile_features(xmem_cross_call_merge_check PRIVATE cxx_std_17)
add_dependencies(xmem_cross_call_merge_check xmem_shim xmem_shim_test_controls)
add_test(NAME xmem_cross_call_merge COMMAND xmem_cross_call_merge_check "$<TARGET_FILE:xmem_shim_test_controls>" check)
add_test(NAME xmem_cross_call_merge_control COMMAND xmem_cross_call_merge_check "$<TARGET_FILE:xmem_shim_test_controls>" control)
add_test(NAME xmem_cross_call_merge_deployed COMMAND xmem_cross_call_merge_check "$<TARGET_FILE:xmem_shim>" deployed)
add_test(NAME xmem_cross_call_merge_default COMMAND xmem_cross_call_merge_check "$<TARGET_FILE:xmem_shim>" default)
add_test(NAME xmem_cross_call_merge_no_join COMMAND xmem_cross_call_merge_check "$<TARGET_FILE:xmem_shim>" no-join)
set_tests_properties(xmem_cross_call_merge xmem_cross_call_merge_control xmem_cross_call_merge_deployed
                     xmem_cross_call_merge_default xmem_cross_call_merge_no_join PROPERTIES TIMEOUT 60)
set_tests_properties(xmem_cross_call_merge_control PROPERTIES ENVIRONMENT "X2W_XMEM_TEST_NO_MERGE=1"
    PASS_REGULAR_EXPRESSION "no-merge control: 32 views" FAIL_REGULAR_EXPRESSION "check_failed")
set_tests_properties(xmem_cross_call_merge_deployed PROPERTIES ENVIRONMENT "X2W_XMEM_TEST_NO_MERGE=1"
    PASS_REGULAR_EXPRESSION "merge: 1 view of 32 pages" FAIL_REGULAR_EXPRESSION "check_failed")
set_tests_properties(xmem_cross_call_merge_default PROPERTIES
    PASS_REGULAR_EXPRESSION "no-merge control: 32 views" FAIL_REGULAR_EXPRESSION "check_failed")
set_tests_properties(xmem_cross_call_merge_no_join PROPERTIES
    PASS_REGULAR_EXPRESSION "no-join: 32 views" FAIL_REGULAR_EXPRESSION "check_failed")
# Placed buffers at view boundaries (tests/xmem_page_runs_device_check.cpp, registered in CMakeLists.txt) with joins
# across mapping calls on: the order of the mapping calls keeps every boundary.
add_test(NAME xmem_page_runs_device_join_warp COMMAND xmem_page_runs_device_check warp join)
set_tests_properties(xmem_page_runs_device_join_warp PROPERTIES TIMEOUT 120)
# tests/xmem_page_runs_check.py expects one view per mapping call: an X2W_XMEM_JOIN_CALLS=1 inherited from the
# caller's environment must not reach it.
if(TEST xmem_shim_page_runs)
    set_property(TEST xmem_shim_page_runs APPEND PROPERTY ENVIRONMENT "X2W_XMEM_JOIN_CALLS=")
endif()

# Slot 31 placeholder split inside an XMem reservation. The control (test-controls build) disables the fallback used
# where the OS does not report placeholders (Wine), so the split fails there; where the OS reports them, the control
# skips (77). The check program sets each mode's test switch itself.
add_executable(xmem_placeholder_split_check tests/xmem_placeholder_split_check.cpp)
target_include_directories(xmem_placeholder_split_check PRIVATE src)
target_compile_features(xmem_placeholder_split_check PRIVATE cxx_std_17)
target_link_libraries(xmem_placeholder_split_check PRIVATE OneCore)
add_dependencies(xmem_placeholder_split_check xmem_shim xmem_shim_test_controls)
set(xmem_placeholder_env "X2W_XMEM_PERMISSIVE=1;X2W_XMEM_GPU_MIRROR=0;X2W_XMEM_LAZY_COMMIT=0;X2W_XMEM_SHARED_VIEWS=0")
add_test(NAME xmem_placeholder_split COMMAND xmem_placeholder_split_check "$<TARGET_FILE:xmem_shim>")
set_tests_properties(xmem_placeholder_split PROPERTIES TIMEOUT 60 ENVIRONMENT "${xmem_placeholder_env}")
# The control passes only when the split itself fails, not on an earlier setup failure.
add_test(NAME xmem_placeholder_split_control COMMAND xmem_placeholder_split_check
         "$<TARGET_FILE:xmem_shim_test_controls>" control)
set_tests_properties(xmem_placeholder_split_control PROPERTIES
    ENVIRONMENT "${xmem_placeholder_env}"
    PASS_REGULAR_EXPRESSION "FAIL: map two pool pages into the middle of the reservation" SKIP_RETURN_CODE 77 TIMEOUT 60)
# The deployed shim with X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1: the switch is ignored and the split succeeds.
add_test(NAME xmem_placeholder_split_deployed COMMAND xmem_placeholder_split_check "$<TARGET_FILE:xmem_shim>" deployed)
set_tests_properties(xmem_placeholder_split_deployed PROPERTIES
    ENVIRONMENT "${xmem_placeholder_env}" SKIP_RETURN_CODE 77 TIMEOUT 60)
# A two-page view over a reserved page and a committed page of one private allocation: the fallback checks every
# region of the view and refuses it (test-controls build: x2w_xmem_alias_counts), and the committed page keeps its
# bytes. The program turns shared views on for this mode.
add_test(NAME xmem_placeholder_split_partial COMMAND xmem_placeholder_split_check
         "$<TARGET_FILE:xmem_shim_test_controls>" partial)
set_tests_properties(xmem_placeholder_split_partial PROPERTIES
    ENVIRONMENT "${xmem_placeholder_env}" SKIP_RETURN_CODE 77 TIMEOUT 60)
add_executable(xmem_map_split_first_check tests/xmem_map_split_first_check.cpp)
target_include_directories(xmem_map_split_first_check PRIVATE src)
target_compile_features(xmem_map_split_first_check PRIVATE cxx_std_17)
add_dependencies(xmem_map_split_first_check xmem_shim xmem_shim_test_controls)
add_test(NAME xmem_map_split_first COMMAND xmem_map_split_first_check "$<TARGET_FILE:xmem_shim>")
add_test(NAME xmem_map_split_first_control COMMAND xmem_map_split_first_check
         "$<TARGET_FILE:xmem_shim_test_controls>" control)
set_tests_properties(xmem_map_split_first PROPERTIES TIMEOUT 60 ENVIRONMENT "${xmem_placeholder_env}")
# The control (test-controls build, X2W_XMEM_TEST_MAP_FIRST=1) passes only when the map-first order is measured
# (2 maps per page), not on a setup failure.
set_tests_properties(xmem_map_split_first_control PROPERTIES
    ENVIRONMENT "${xmem_placeholder_env}"
    PASS_REGULAR_EXPRESSION "map-first control: maps=2N" TIMEOUT 60)
# The deployed shim with X2W_XMEM_TEST_MAP_FIRST=1: the switch is ignored, so the split-first counts hold.
add_test(NAME xmem_map_split_first_deployed COMMAND xmem_map_split_first_check "$<TARGET_FILE:xmem_shim>" deployed)
set_tests_properties(xmem_map_split_first_deployed PROPERTIES TIMEOUT 60 ENVIRONMENT "${xmem_placeholder_env}")

# A request whose destination page an XMem run break holds unmapped (the xmem test-controls build's
# X2W_XMEM_TEST_HOLD_WINDOW) must not complete within the 1000 ms hold, then land after the release.
# Control: XCRDAPI without the x2w_xmem_copy_to_destination lookup judges the destination with its own VirtualQuery
# inside the window and refuses it.
add_executable(startup_dstorage_break_window_check tests/startup_dstorage_break_window_check.cpp)
target_compile_features(startup_dstorage_break_window_check PRIVATE cxx_std_17)
add_dependencies(startup_dstorage_break_window_check startup_stub_xcrdapi xmem_shim_test_controls)
add_test(NAME startup_dstorage_break_window COMMAND startup_dstorage_break_window_check
         "$<TARGET_FILE:xmem_shim_test_controls>" "$<TARGET_FILE:startup_stub_xcrdapi>")
set_tests_properties(startup_dstorage_break_window PROPERTIES TIMEOUT 60)
# Which copies take xmem's mapping mutex (x2w_xmem_copy_counts): with shared views, a VirtualAlloc destination copies
# without it and one in a run with it (plain); with shared views off, an XMem page destination copies without it.
add_test(NAME startup_dstorage_break_window_plain COMMAND startup_dstorage_break_window_check
         "$<TARGET_FILE:xmem_shim_test_controls>" "$<TARGET_FILE:startup_stub_xcrdapi>" plain)
add_test(NAME startup_dstorage_break_window_views_off COMMAND startup_dstorage_break_window_check
         "$<TARGET_FILE:xmem_shim_test_controls>" "$<TARGET_FILE:startup_stub_xcrdapi>" views-off)
# XCRDAPI's logged outcome for each copy error, forced by the test-controls build (X2W_XMEM_TEST_COPY_ERROR).
add_test(NAME startup_dstorage_break_window_outcomes COMMAND startup_dstorage_break_window_check
         "$<TARGET_FILE:xmem_shim_test_controls>" "$<TARGET_FILE:startup_stub_xcrdapi>" outcomes)
set_tests_properties(startup_dstorage_break_window_plain startup_dstorage_break_window_views_off
                     startup_dstorage_break_window_outcomes PROPERTIES TIMEOUT 60)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/startup_stubs/xcrdapi.cpp)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/startup_stubs/xcrdapi.cpp" xcrdapi_text)
string(REPLACE "\"x2w_xmem_copy_to_destination\"" "\"x2w_xmem_copy_to_destination_control\"" xcrdapi_control_text
       "${xcrdapi_text}")
if(xcrdapi_control_text STREQUAL xcrdapi_text)
    message(FATAL_ERROR "xcrdapi.cpp lost the anchor used by startup_dstorage_break_window_control")
endif()
set(xcrdapi_control "${CMAKE_CURRENT_BINARY_DIR}/xcrdapi_break_window_control/xcrdapi.cpp")
file(WRITE "${xcrdapi_control}" "${xcrdapi_control_text}")
add_library(startup_stub_xcrdapi_break_window_control SHARED "${xcrdapi_control}" src/startup_stubs/xcrdapi.def)
set_target_properties(startup_stub_xcrdapi_break_window_control PROPERTIES OUTPUT_NAME XCRDAPI
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xcrdapi-break-window-control"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xcrdapi-break-window-control"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xcrdapi-break-window-control"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_include_directories(startup_stub_xcrdapi_break_window_control PRIVATE src)
target_compile_features(startup_stub_xcrdapi_break_window_control PRIVATE cxx_std_17)
target_link_libraries(startup_stub_xcrdapi_break_window_control PRIVATE psapi)
add_dependencies(startup_dstorage_break_window_check startup_stub_xcrdapi_break_window_control)
add_test(NAME startup_dstorage_break_window_control COMMAND startup_dstorage_break_window_check
         "$<TARGET_FILE:xmem_shim_test_controls>" "$<TARGET_FILE:startup_stub_xcrdapi_break_window_control>")
set_tests_properties(startup_dstorage_break_window_control PROPERTIES TIMEOUT 60
    PASS_REGULAR_EXPRESSION "FAIL break window: refused while the run was broken")
