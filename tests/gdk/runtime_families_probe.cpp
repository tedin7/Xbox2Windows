// Original fixture for tools/gdk_runtime_oracle.py. It prints only public
// observations from the desktop GDK runtime; it never prints package contents.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <XGameRuntime.h>

static void hr(const char* name, HRESULT value)
{
    std::printf("%s=0x%08X\n", name, static_cast<unsigned>(value));
}

int main()
{
#ifndef X2W_SHIM
    hr("init", XGameRuntimeInitialize());
#endif

    std::printf("device=%u\n", static_cast<unsigned>(XSystemGetDeviceType()));

    char console_id[128]{};
    std::size_t console_used = 0;
    hr("console_id", XSystemGetConsoleId(sizeof(console_id), console_id, &console_used));
    std::printf("console_used=%zu console_value=%s\n", console_used, console_id);

    std::uint32_t title_id = 0;
    hr("title_id", XGameGetXboxTitleId(&title_id));
    std::printf("title_value=0x%08X\n", static_cast<unsigned>(title_id));

    std::size_t path_size = 0;
    hr("pls_path_size", XPersistentLocalStorageGetPathSize(&path_size));
    std::printf("pls_path_size_value=%zu\n", path_size);
    if (path_size > 0)
    {
        std::vector<char> path(path_size + 1, 0);
        std::size_t used = 0;
        hr("pls_path", XPersistentLocalStorageGetPath(path.size(), path.data(), &used));
        std::printf("pls_path_used=%zu pls_path_value=%s\n", used, path.data());
    }

    XPersistentLocalStorageSpaceInfo space{};
    hr("pls_space", XPersistentLocalStorageGetSpaceInfo(&space));
    std::printf("pls_space_available=%llu total_free=%llu used=%llu total=%llu\n",
                static_cast<unsigned long long>(space.availableFreeBytes),
                static_cast<unsigned long long>(space.totalFreeBytes),
                static_cast<unsigned long long>(space.usedBytes),
                static_cast<unsigned long long>(space.totalBytes));

    XPackageMountHandle mount = nullptr;
    hr("xpackage_mount_missing", XPackageMount("x2w-nonexistent-package", &mount));
    std::printf("xpackage_mount_handle=%p\n", static_cast<void*>(mount));
    if (mount)
    {
        std::size_t mount_size = 0;
        hr("xpackage_path_size", XPackageGetMountPathSize(mount, &mount_size));
        std::printf("xpackage_path_size_value=%zu\n", mount_size);
        if (mount_size > 0)
        {
            std::vector<char> path(mount_size + 1, 0);
            hr("xpackage_path", XPackageGetMountPath(mount, path.size(), path.data()));
            std::printf("xpackage_path_value=%s\n", path.data());
        }
        XPackageCloseMountHandle(mount);
        std::printf("xpackage_closed=1\n");
    }

    XTaskQueueHandle queue = nullptr;
    hr("queue_create", XTaskQueueCreate(XTaskQueueDispatchMode::Manual,
                                        XTaskQueueDispatchMode::Manual, &queue));
    std::printf("queue_handle=%p\n", static_cast<void*>(queue));
    if (queue)
    {
        std::printf("queue_dispatch=%u\n",
                    XTaskQueueDispatch(queue, XTaskQueuePort::Completion, 0) ? 1u : 0u);
        XTaskQueueCloseHandle(queue);
        std::printf("queue_closed=1\n");
    }

#ifndef X2W_SHIM
    char sandbox[64]{};
    std::size_t sandbox_used = 0;
    hr("sandbox", XSystemGetXboxLiveSandboxId(sizeof(sandbox), sandbox, &sandbox_used));
    std::printf("sandbox_used=%zu sandbox_value=%s\n", sandbox_used, sandbox);
    XGameRuntimeUninitialize();
#endif
    return 0;
}
