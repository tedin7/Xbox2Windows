# Slot 133 existing-heap cap (opt-in X2W_D3D12_CAP_EXISTING_HEAP=1, <= 256 MiB), deferred
# commit_lazy_block, and incremental reserve_upload_ring growth (Refs #1199, Fixes #2765).
# The default cases exercise the aliased >256 MiB placement and its late-write fenced readback;
# the capped cases exercise the opt-in cap's copied fallback, the <= 256 MiB alias cases and the
# upload-ring growth.
add_executable(startup_slot133_heap_cap_check tests/startup_slot133_heap_cap_check.cpp)
target_include_directories(startup_slot133_heap_cap_check PRIVATE src)
target_compile_features(startup_slot133_heap_cap_check PRIVATE cxx_std_17)
target_link_libraries(startup_slot133_heap_cap_check PRIVATE d3d12 dxgi)
add_dependencies(startup_slot133_heap_cap_check xmem_shim d3d12_xs_shim)

foreach(backend IN ITEMS warp hardware)
    foreach(debug IN ITEMS 0 1)
        set(case startup_slot133_heap_cap_${backend}_debug${debug})
        add_test(NAME ${case} COMMAND startup_slot133_heap_cap_check ${backend})
        set_tests_properties(${case} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
            ENVIRONMENT "X2W_D3D12_DEBUG=${debug}")
    endforeach()
endforeach()

foreach(backend IN ITEMS warp hardware)
    set(case startup_slot133_heap_cap_capped_${backend})
    add_test(NAME ${case} COMMAND startup_slot133_heap_cap_check ${backend} cap)
    set_tests_properties(${case} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        ENVIRONMENT "X2W_D3D12_DEBUG=0")
endforeach()

# Video-sized CPU-written textures (1 MiB and up) are filled from their own upload buffer, not the shared ring (the audit's
# black-screen cause: the intro video's plane uploads filled the ring). Forty 1920x1080 RGBA8 textures, read back exact.
foreach(backend IN ITEMS warp hardware)
    set(case startup_slot133_video_upload_${backend})
    add_test(NAME ${case} COMMAND startup_slot133_heap_cap_check ${backend} video)
    set_tests_properties(${case} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
        ENVIRONMENT "X2W_D3D12_DEBUG=0")
endforeach()
