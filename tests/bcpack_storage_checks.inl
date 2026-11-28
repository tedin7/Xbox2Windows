// Included by the source-built startup consumer after its queue/zlib helpers.
// SPDX-License-Identifier: MIT

bool bcpack_storage_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");
    const auto library = LoadLibraryW(L"XCRDAPI.dll");
    if (!library) return false;
    const auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(library,"DStorageGetFactory"));
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr,&factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);
    for (unsigned fmt : {2u,6u}) {
        const auto s = bcpack_fixture::random_symbols(fmt,4,0x7700+fmt);
        std::vector<uint8_t> packed;
        if (!bcpack_fixture::encode(fmt,s,4,packed)) return false;
        const auto zipped = make_stored_zlib(packed.data(),packed.size());
        const auto* expected = fmt == 2 ? BC2_STORAGE_EXPECTED : BC6_STORAGE_EXPECTED;
        // Memory/file x raw/zlib, linear/page destinations; then failures followed by valid requests.
        for (unsigned scenario = 0; scenario < (fmt == 6 ? 14u : 12u); ++scenario) {
            const bool file = (scenario & 1) != 0;
            const bool zlib = (scenario & 2) != 0;
            const bool page = (scenario >= 4 && scenario < 10) || scenario == 13;
            const bool invalid = scenario >= 8;
            std::vector<uint8_t> damaged;
            if (scenario >= 12) {
                auto bad = s;
                bad[15*8+6] = 14; // final block's undefined mode, after three valid blocks
                if (!bcpack_fixture::encode(fmt,bad,4,damaged)) return false;
            }
            const auto& input = scenario >= 12 ? damaged : zlib ? zipped : packed;
            void *queue_ptr = nullptr, *status_ptr = nullptr;
            uint8_t desc[0x20] = {}; *reinterpret_cast<uint16_t*>(desc+8) = 16;
            if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory,desc,nullptr,&queue_ptr) != 0 || !queue_ptr ||
                reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory,2,"BCPack",nullptr,&status_ptr) != 0 || !status_ptr)
                return false;
            auto* queue = static_cast<Object*>(queue_ptr);
            auto* status = static_cast<Object*>(status_ptr);
            std::vector<uint8_t> linear(96,0xa5), following(64,0xa5);
            size_t page_count = 2;
            u64 pages[2] = {};
            ULONG_PTR tokens[2] = {};
            void* view = nullptr;
            if (page) {
                if (!XMemAllocatePhysicalPages(0x20400000ull,&page_count,pages) || page_count != 2) return false;
                tokens[0] = pages[0]; tokens[1] = pages[1];
                view = XMemVirtualAlloc(nullptr,131072,0x20402000,0x21,PAGE_READWRITE);
                if (!view || XMemMapPhysicalPages(view,2,tokens) != view) return false;
                std::memset(view,0xa5,131072);
                if (scenario == 8 || scenario == 9) tokens[1] = 0; // missing second page: no partial publication
            }
            wchar_t temp_dir[MAX_PATH], temp_file[MAX_PATH] = {};
            Object* source_file = nullptr;
            if (file) {
                if (!GetTempPathW(MAX_PATH,temp_dir) || !GetTempFileNameW(temp_dir,L"bcp",0,temp_file)) return false;
                const auto h = CreateFileW(temp_file,GENERIC_WRITE,0,nullptr,CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
                if (h == INVALID_HANDLE_VALUE) return false;
                const uint8_t prefix[13] = {};
                DWORD written = 0;
                const bool wrote = WriteFile(h,prefix,sizeof(prefix),&written,nullptr) && written == sizeof(prefix) &&
                    WriteFile(h,input.data(),static_cast<DWORD>(input.size()),&written,nullptr) && written == input.size();
                CloseHandle(h);
                void* ptr = nullptr;
                if (!wrote || reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory,temp_file,nullptr,&ptr) != 0 || !ptr) return false;
                source_file = static_cast<Object*>(ptr);
            }
            uint8_t requests[2][0x48] = {};
            for (unsigned k = 0; k < 2; ++k) {
                auto* r = requests[k];
                const bool first = k == 0;
                const bool use_file = first && file, use_zlib = first && zlib, use_pages = first && page;
                const auto& src = first ? input : packed;
                unsigned request_format = first && (scenario == 10 || scenario == 11) ? 8 : fmt;
                *reinterpret_cast<u64*>(r) = (request_format << 1) | (use_zlib ? 1 : 0) |
                    (use_file ? 0 : 0x400000) | (use_pages ? 1ull<<21 : 0);
                *reinterpret_cast<void**>(r+0x08) = use_pages ? static_cast<void*>(tokens) :
                    first ? static_cast<void*>(linear.data()+16) : static_cast<void*>(following.data());
                *reinterpret_cast<u64*>(r+0x10) = use_pages ? 65520 : 0;
                *reinterpret_cast<unsigned*>(r+0x18) = 64;
                *reinterpret_cast<const void**>(r+0x20) = use_file ? static_cast<const void*>(source_file) : src.data();
                *reinterpret_cast<u64*>(r+0x28) = use_file ? 13 : 0;
                *reinterpret_cast<unsigned*>(r+0x30) = static_cast<unsigned>(src.size());
                *reinterpret_cast<unsigned*>(r+0x34) = static_cast<unsigned>(packed.size());
                reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue,r);
                reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue,status,k);
            }
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
            for (int i = 0; i < 200 && !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status,1); ++i) Sleep(10);
            bool ok = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status,1);
            const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status,0);
            ok = ok && (invalid ? hr != 0 : hr == 0) &&
                reinterpret_cast<GetHResultFn>(status->vtable[4])(status,1) == 0 && std::memcmp(following.data(),expected,64) == 0;
            const auto* destination = page ? static_cast<const uint8_t*>(view) : linear.data();
            const size_t size = page ? 131072 : linear.size(), offset = page ? 65520 : 16;
            for (size_t at = 0; at < size; ++at) {
                const uint8_t wanted = !invalid && at >= offset && at < offset+64 ? expected[at-offset] : 0xa5;
                if (destination[at] != wanted) ok = false;
            }
            if (source_file) {
                reinterpret_cast<CloseFileFn>(source_file->vtable[3])(source_file);
                DeleteFileW(temp_file);
            }
            if (view) {
                XMemMapPhysicalPages(view,2,nullptr);
                VirtualFree(view,0,MEM_RELEASE);
                XMemFreePhysicalPages(2,pages);
            }
            std::printf("BC%u storage scenario %u: %s\n",fmt,scenario,ok ? "ok" : "FAIL");
            if (!ok) return false;
        }
    }
    return true;
}
