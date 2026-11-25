// Vulkan host for translated pixel shaders. It mirrors draw_verified (tests/pixel_aot_render.cpp) on D3D12: a
// full-screen triangle (3 vertices, no vertex buffer), an RGBA8_UNORM texture sampled with point/clamp, 256 bytes
// of constants, an RGBA32F (or RGBA32UI) target cleared to -1, then a raw readback of the pixels into OUT.
//
// Expected bindings (dxc -fvk-b-shift 0 0 -fvk-t-shift 16 0 -fvk-s-shift 32 0 -fvk-u-shift 48 0): set 0,
// b0 = 0, t0 = 16, t1 = 17, s0 = 32. Compile the vertex shader with -fvk-invert-y to keep the D3D12 orientation.
//
// Usage: vk_pixel_draw DEVICE VS.spv PS.spv CONSTANTS.bin TEX.rgba TW TH OUT.bin RW RH float|uint [TEX2.rgba]
//        vk_pixel_draw --caps DEVICE
//        vk_pixel_draw --draw DEVICE VS.spv PS.spv OUT.bin [options]
// The --draw form is the generic draw driven by tests/vk_offscreen.py. Options (all optional except --target):
//   --target W H float|uint   render target extent and format (RGBA32F or RGBA32UI)
//   --constants FILE          256 bytes bound at b0 (default zeros)
//   --extra FILE              512 bytes: the 256-byte buffers bound at b1 then b2 (default zeros)
//   --texture W H FILE...     RGBA8_UNORM texture at t0, one file per mip level (level k is max(1, W >> k) wide
//                             and high); t1 is a copy of it. Default is one zero texel.
//   --viewport X Y W H        viewport in target pixels (default the whole target); the scissor stays whole
//   --indices A,B,C           draw three indexed vertices instead of vertices 0, 1 and 2
//   --write-mask N            color write mask, bit 0 red (default 15)
//   --clear A,B,C,D           clear color: floats for float, unsigned integers for uint (default -1 or 0xffffffff)
//   --allow-denorm-mismatch   run a module that declares DenormPreserve 32 on a device without
//                             shaderDenormPreserveFloat32 (otherwise exit 77)
//   DEVICE: substring of the GPU name (for example Intel, llvmpipe).
// Both forms print one "device=..." line with the driver and float-control properties on stdout.
// Exit code: 0 drawn (or --caps answered), 77 no matching GPU or missing capability, 78 the driver refused the
// shader module or the pipeline (validation refusal), 1 error.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice phys;

static void *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(n ? n : 1);
    if (fread(data, 1, n, f) != (size_t)n) { perror(path); exit(1); }
    fclose(f);
    *size = n;
    return data;
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; } Buffer;

static Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer b;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    CHECK(vkCreateBuffer(dev, &info, NULL, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
        .memoryTypeIndex = memory_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CHECK(vkAllocateMemory(dev, &alloc, NULL, &b.mem));
    CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

typedef struct { VkImage img; VkDeviceMemory mem; VkImageView view; } Image;

static Image make_image(uint32_t w, uint32_t h, uint32_t levels, VkFormat format, VkImageUsageFlags usage)
{
    Image im;
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
        .extent = {w, h, 1}, .mipLevels = levels, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    CHECK(vkCreateImage(dev, &info, NULL, &im.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, im.img, &req);
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
        .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CHECK(vkAllocateMemory(dev, &alloc, NULL, &im.mem));
    CHECK(vkBindImageMemory(dev, im.img, im.mem, 0));
    VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1}};
    CHECK(vkCreateImageView(dev, &view, NULL, &im.view));
    return im;
}

static void barrier(VkCommandBuffer cmd, VkImage img, uint32_t levels, VkImageLayout from, VkImageLayout to,
                    VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = src, .dstAccessMask = dst,
        .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1}};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

static VkShaderModule module(const char *path)
{
    size_t size;
    void *code = read_file(path, &size);
    VkShaderModuleCreateInfo info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &info, NULL, &m);
    if (r != VK_SUCCESS) { fprintf(stderr, "refused: vkCreateShaderModule(%s) -> %d\n", path, r); exit(78); }
    free(code);
    return m;
}

// The staging buffer holds the mip levels one after the other, level 0 first, four bytes per texel.
static void upload_texture(VkCommandBuffer cmd, Image *tex, Buffer *staging, uint32_t w, uint32_t h, uint32_t levels)
{
    barrier(cmd, tex->img, levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkDeviceSize offset = 0;
    for (uint32_t level = 0; level < levels; ++level) {
        const uint32_t lw = w >> level ? w >> level : 1, lh = h >> level ? h >> level : 1;
        VkBufferImageCopy copy = {.bufferOffset = offset,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1}, .imageExtent = {lw, lh, 1}};
        vkCmdCopyBufferToImage(cmd, staging->buf, tex->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        offset += (VkDeviceSize)lw * lh * 4;
    }
    barrier(cmd, tex->img, levels, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

// Whether the module declares OpExecutionMode DenormPreserve with width 32 (SPIR-V float controls).
static int declares_denorm_preserve32(const char *path)
{
    size_t size;
    const uint32_t *w = read_file(path, &size);
    const size_t n = size / 4;
    for (size_t i = 5; i < n;) {
        const uint32_t count = w[i] >> 16, op = w[i] & 0xffff;
        if (!count || i + count > n) break;
        if (op == 16 && count >= 4 && w[i + 2] == 4459 && w[i + 3] == 32) return 1;  // OpExecutionMode
        i += count;
    }
    return 0;
}

static int parse_clear(const char *text, int integer, uint32_t out[4])
{
    char *end;
    for (int i = 0; i < 4; ++i) {
        if (integer) out[i] = (uint32_t)strtoul(text, &end, 0);
        else { float f = strtof(text, &end); memcpy(&out[i], &f, 4); }
        if (end == text || (i < 3 ? *end != ',' : *end != 0)) return 0;
        text = end + 1;
    }
    return 1;
}

int main(int argc, char **argv)
{
    const int caps_only = argc == 3 && !strcmp(argv[1], "--caps");
    const int draw_mode = argc >= 2 && !strcmp(argv[1], "--draw");
    if (!caps_only && !draw_mode && argc != 12 && argc != 13) {
        fprintf(stderr, "usage: %s DEVICE VS.spv PS.spv CONSTANTS.bin TEX.rgba TW TH OUT.bin RW RH float|uint "
                        "[TEX2.rgba]\n       %s --caps DEVICE\n       %s --draw DEVICE VS.spv PS.spv OUT.bin "
                        "--target W H float|uint [options]\n", argv[0], argv[0], argv[0]);
        return 1;
    }
    const char *want, *vs_path = NULL, *ps_path = NULL, *out_path = NULL;
    uint32_t tw = 0, th = 0, rw = 0, rh = 0, levels = 1, write_mask = 0xf;
    int integer = 0, indexed = 0, have_clear = 0, allow_denorm_mismatch = 0;
    const char *clear_text = NULL;
    float viewport_rect[4] = {0, 0, 0, 0};
    int have_viewport = 0;
    uint32_t indices[3] = {0, 1, 2}, clear_bits[4] = {0, 0, 0, 0};
    const char *constants_path = NULL, *extra_path = NULL, *level_paths[16] = {0}, *tex2_path = NULL;
    if (caps_only) {
        want = argv[2];
    } else if (draw_mode) {
        if (argc < 6) { fprintf(stderr, "--draw needs DEVICE VS.spv PS.spv OUT.bin\n"); return 1; }
        want = argv[2]; vs_path = argv[3]; ps_path = argv[4]; out_path = argv[5];
        int have_target = 0, have_texture = 0;
        for (int i = 6; i < argc; ++i) {
            const char *opt = argv[i];
#define NEED(n) do { if (i + (n) >= argc) { fprintf(stderr, "%s needs %d values\n", opt, (n)); return 1; } } while (0)
            if (!strcmp(opt, "--target")) {
                NEED(3); rw = atoi(argv[i + 1]); rh = atoi(argv[i + 2]);
                if (strcmp(argv[i + 3], "float") && strcmp(argv[i + 3], "uint")) { fprintf(stderr, "bad target format\n"); return 1; }
                integer = !strcmp(argv[i + 3], "uint"); have_target = 1; i += 3;
            } else if (!strcmp(opt, "--constants")) { NEED(1); constants_path = argv[++i];
            } else if (!strcmp(opt, "--extra")) { NEED(1); extra_path = argv[++i];
            } else if (!strcmp(opt, "--texture")) {
                NEED(3); tw = atoi(argv[i + 1]); th = atoi(argv[i + 2]); i += 2;
                for (levels = 0; i + 1 < argc && strncmp(argv[i + 1], "--", 2) && levels < 16; ) level_paths[levels++] = argv[++i];
                if (!tw || !th || !levels) { fprintf(stderr, "--texture needs W H and at least one level file\n"); return 1; }
                have_texture = 1;
            } else if (!strcmp(opt, "--viewport")) {
                NEED(4); for (int k = 0; k < 4; ++k) viewport_rect[k] = strtof(argv[i + 1 + k], NULL);
                have_viewport = 1; i += 4;
            } else if (!strcmp(opt, "--indices")) {
                NEED(1); if (sscanf(argv[++i], "%u,%u,%u", &indices[0], &indices[1], &indices[2]) != 3) { fprintf(stderr, "bad --indices\n"); return 1; }
                indexed = 1;
            } else if (!strcmp(opt, "--write-mask")) { NEED(1); write_mask = (uint32_t)strtoul(argv[++i], NULL, 0);
            } else if (!strcmp(opt, "--clear")) { NEED(1); clear_text = argv[++i]; have_clear = 1;
            } else if (!strcmp(opt, "--allow-denorm-mismatch")) { allow_denorm_mismatch = 1;
            } else { fprintf(stderr, "unknown option %s\n", opt); return 1; }
#undef NEED
        }
        if (!have_target || !rw || !rh) { fprintf(stderr, "--draw needs --target W H float|uint\n"); return 1; }
        if (!have_texture) { tw = th = 1; levels = 1; }
        if (have_clear && !parse_clear(clear_text, integer, clear_bits)) { fprintf(stderr, "bad --clear\n"); return 1; }
    } else {
        want = argv[1]; vs_path = argv[2]; ps_path = argv[3]; constants_path = argv[4]; level_paths[0] = argv[5];
        tw = atoi(argv[6]); th = atoi(argv[7]); out_path = argv[8]; rw = atoi(argv[9]); rh = atoi(argv[10]);
        integer = !strcmp(argv[11], "uint");
        tex2_path = argc == 13 ? argv[12] : NULL;
    }
    const VkFormat target_format = integer ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT;
    size_t constants_size = 0, extra_size = 0, tex_size = 0, tex2_size = 0;
    void *constants = NULL, *extra = NULL, *texels = NULL, *texels2 = NULL;
    if (!caps_only) {
        constants = constants_path ? read_file(constants_path, &constants_size) : calloc(1, 256);
        if (!constants_path) constants_size = 256;
        extra = extra_path ? read_file(extra_path, &extra_size) : calloc(1, 512);
        if (!extra_path) extra_size = 512;
        size_t level_total = 0;
        for (uint32_t l = 0; l < levels; ++l) {
            const uint32_t lw = tw >> l ? tw >> l : 1, lh = th >> l ? th >> l : 1;
            level_total += (size_t)lw * lh * 4;
        }
        if (draw_mode && !level_paths[0]) {
            texels = calloc(1, 4); tex_size = 4;
        } else {
            texels = malloc(level_total ? level_total : 1);
            size_t at = 0;
            for (uint32_t l = 0; l < levels; ++l) {
                const uint32_t lw = tw >> l ? tw >> l : 1, lh = th >> l ? th >> l : 1;
                size_t n;
                void *level = read_file(level_paths[l], &n);
                if (n != (size_t)lw * lh * 4) {
                    fprintf(stderr, "expected %u x %u RGBA8 texels in level %u\n", lw, lh, l);
                    return 1;
                }
                memcpy((char *)texels + at, level, n); at += n; free(level);
            }
            tex_size = level_total;
        }
        texels2 = tex2_path ? read_file(tex2_path, &tex2_size) : NULL;
        if (constants_size != 256 || extra_size != 512 || (texels2 && tex2_size != tex_size)) {
            fprintf(stderr, "expected 256 constant bytes, 512 extra constant bytes and %u x %u RGBA8 texels\n", tw, th);
            return 1;
        }
    }

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    CHECK(vkCreateInstance(&inst_info, NULL, &inst));
    uint32_t count = 16;
    VkPhysicalDevice devices[16];
    CHECK(vkEnumeratePhysicalDevices(inst, &count, devices));
    VkPhysicalDeviceProperties props;
    phys = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count && !phys; ++i) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (strstr(props.deviceName, want)) phys = devices[i];
    }
    if (!phys) { fprintf(stderr, "no Vulkan device matching '%s'\n", want); return 77; }
    if (props.apiVersion < VK_API_VERSION_1_2) { fprintf(stderr, "Vulkan 1.2 required\n"); return 77; }

    // Enable every feature the driver reports (16-bit types and so on) without requiring any.
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11};
    vkGetPhysicalDeviceFeatures2(phys, &features);
    VkPhysicalDeviceVulkan12Properties p12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p12};
    vkGetPhysicalDeviceProperties2(phys, &props2);
    printf("device=\"%s\" api=%u.%u.%u driver=\"%s\" driver_info=\"%s\" driver_version=%08x "
           "denorm_preserve32=%u denorm_flush_to_zero32=%u signed_zero_inf_nan32=%u\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
           p12.driverName, p12.driverInfo, props.driverVersion, p12.shaderDenormPreserveFloat32,
           p12.shaderDenormFlushToZeroFloat32, p12.shaderSignedZeroInfNanPreserveFloat32);
    if (caps_only) return 0;
    if (draw_mode && !p12.shaderDenormPreserveFloat32 &&
        (declares_denorm_preserve32(vs_path) || declares_denorm_preserve32(ps_path))) {
        if (!allow_denorm_mismatch) {
            fprintf(stderr, "capability skip: the module declares DenormPreserve 32 and the device lacks "
                            "shaderDenormPreserveFloat32\n");
            return 77;
        }
        fprintf(stderr, "note: the module declares DenormPreserve 32 and the device lacks "
                        "shaderDenormPreserveFloat32; drawing anyway (diagnostic run)\n");
    }

    uint32_t qcount = 8;
    VkQueueFamilyProperties qprops[8];
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, qprops);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < qcount && family == UINT32_MAX; ++i)
        if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) family = i;
    if (family == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 77; }
    const float priority = 1;
    VkDeviceQueueCreateInfo qinfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
        .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo dinfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qinfo};
    CHECK(vkCreateDevice(phys, &dinfo, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);

    // Resources.
    Buffer ubo = make_buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    memcpy(ubo.map, constants, 256);
    Buffer ubo1 = make_buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    memcpy(ubo1.map, extra, 256);
    Buffer ubo2 = make_buffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    memcpy(ubo2.map, (char *)extra + 256, 256);
    Buffer index_buffer = make_buffer(12, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    memcpy(index_buffer.map, indices, 12);
    Buffer staging = make_buffer(tex_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    memcpy(staging.map, texels, tex_size);
    Buffer staging2 = make_buffer(tex_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    memcpy(staging2.map, texels2 ? texels2 : texels, tex_size);
    const VkImageUsageFlags sampled = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Image tex = make_image(tw, th, levels, VK_FORMAT_R8G8B8A8_UNORM, sampled);
    Image tex2 = make_image(tw, th, levels, VK_FORMAT_R8G8B8A8_UNORM, sampled);
    Image target = make_image(rw, rh, 1, target_format,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const VkDeviceSize out_size = (VkDeviceSize)rw * rh * 16;
    Buffer readback = make_buffer(out_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkSamplerCreateInfo sinfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = VK_LOD_CLAMP_NONE};
    VkSampler sampler;
    CHECK(vkCreateSampler(dev, &sinfo, NULL, &sampler));

    // Descriptors: b0 = 0, b1 = 1, b2 = 2, t0 = 16, t1 = 17, s0 = 32.
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {32, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
    };
    VkDescriptorSetLayoutCreateInfo linfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 6, .pBindings = bindings};
    VkDescriptorSetLayout set_layout;
    CHECK(vkCreateDescriptorSetLayout(dev, &linfo, NULL, &set_layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
                                    {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo pinfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
        .poolSizeCount = 3, .pPoolSizes = sizes};
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(dev, &pinfo, NULL, &pool));
    VkDescriptorSetAllocateInfo ainfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
        .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(dev, &ainfo, &set));
    VkDescriptorBufferInfo ubo_info = {ubo.buf, 0, 256}, ubo1_info = {ubo1.buf, 0, 256}, ubo2_info = {ubo2.buf, 0, 256};
    VkDescriptorImageInfo tex_info = {.imageView = tex.view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo tex2_info = {.imageView = tex2.view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo samp_info = {.sampler = sampler};
    VkWriteDescriptorSet writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &ubo_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &ubo1_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &ubo2_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 16, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &tex_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 17, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &tex2_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 32, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &samp_info},
    };
    vkUpdateDescriptorSets(dev, 6, writes, 0, NULL);

    // Pipeline.
    VkAttachmentDescription att = {.format = target_format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
    VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
        .pColorAttachments = &ref};
    VkSubpassDependency dep = {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, 0};
    VkRenderPassCreateInfo rpinfo = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
        .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sub, .dependencyCount = 1, .pDependencies = &dep};
    VkRenderPass rp;
    CHECK(vkCreateRenderPass(dev, &rpinfo, NULL, &rp));
    VkFramebufferCreateInfo fbinfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
        .attachmentCount = 1, .pAttachments = &target.view, .width = rw, .height = rh, .layers = 1};
    VkFramebuffer fb;
    CHECK(vkCreateFramebuffer(dev, &fbinfo, NULL, &fb));
    VkPipelineLayoutCreateInfo plinfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
        .pSetLayouts = &set_layout};
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(dev, &plinfo, NULL, &layout));
    VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = module(vs_path), .pName = "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = module(ps_path), .pName = "main"},
    };
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport viewport = have_viewport
        ? (VkViewport){viewport_rect[0], viewport_rect[1], viewport_rect[2], viewport_rect[3], 0, 1}
        : (VkViewport){0, 0, (float)rw, (float)rh, 0, 1};
    VkRect2D scissor = {{0, 0}, {rw, rh}};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = write_mask};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend};
    VkGraphicsPipelineCreateInfo gpinfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
        .pStages = stages, .pVertexInputState = &vin, .pInputAssemblyState = &ia, .pViewportState = &vp,
        .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = layout,
        .renderPass = rp};
    VkPipeline pipeline;
    VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpinfo, NULL, &pipeline);
    if (pr != VK_SUCCESS) { fprintf(stderr, "refused: vkCreateGraphicsPipelines -> %d\n", pr); return 78; }

    // Commands.
    VkCommandPoolCreateInfo cpinfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
    VkCommandPool cpool;
    CHECK(vkCreateCommandPool(dev, &cpinfo, NULL, &cpool));
    VkCommandBufferAllocateInfo cbinfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cbinfo, &cmd));
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CHECK(vkBeginCommandBuffer(cmd, &begin));
    upload_texture(cmd, &tex, &staging, tw, th, levels);
    upload_texture(cmd, &tex2, &staging2, tw, th, levels);
    VkClearValue clear;
    if (have_clear) memcpy(clear.color.uint32, clear_bits, sizeof(clear_bits));
    else if (integer) for (int i = 0; i < 4; ++i) clear.color.uint32[i] = 0xffffffffu;
    else for (int i = 0; i < 4; ++i) clear.color.float32[i] = -1.0f;
    VkRenderPassBeginInfo rpbegin = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
        .renderArea = {{0, 0}, {rw, rh}}, .clearValueCount = 1, .pClearValues = &clear};
    vkCmdBeginRenderPass(cmd, &rpbegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    if (indexed) {
        vkCmdBindIndexBuffer(cmd, index_buffer.buf, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, 3, 1, 0, 0, 0);
    } else {
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {rw, rh, 1}};
    vkCmdCopyImageToBuffer(cmd, target.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &copy);
    VkBufferMemoryBarrier host = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = readback.buf, .size = VK_WHOLE_SIZE};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &host, 0, NULL);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));

    FILE *out = fopen(out_path, "wb");
    if (!out || fwrite(readback.map, 1, out_size, out) != out_size || fclose(out)) { perror(out_path); return 1; }
    printf("pixels=%u\n", rw * rh);
    return 0;
}
