// src/startup_stubs/lock_sites.h: the draw_lock top-holders line. Sites are recorded by function and line, the line lists the
// biggest holders first with non-zero hold counts, and the table starts empty again after a line.
// --mutate skip-held: the release never records its hold (the failure that kept the line from showing holds): every check must fail.
#include "startup_stubs/lock_sites.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}
const char site_frequent[] = "frequent";
const char site_slow[] = "slow_rare";
}  // namespace

int main(int argc, char** argv)
{
    const bool skip_held = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "skip-held");
    x2w_local::LockSites sites;
    char line[1024];
    check("an empty table prints nothing", !sites.top_line(line, sizeof(line), 1000.0));

    for (int i = 0; i < 500; ++i) {  // 500 short holds: 0.1 ms each
        sites.waited(site_frequent, 10, 100);
        if (!skip_held) sites.held(site_frequent, 10, 100);
    }
    for (int i = 0; i < 3; ++i) {  // 3 long holds: 20 ms each
        sites.waited(site_slow, 20, 5000);
        if (!skip_held) sites.held(site_slow, 20, 20000);
    }
    if (!skip_held) sites.held(site_frequent, 11, 50);  // a second line of the same function is its own site
    check("the table has something to print", sites.top_line(line, sizeof(line), 1000.0));
    const std::string text = line;
    check("the line starts with the heartbeat prefix", text.rfind("x2w-render draw_lock top holders", 0) == 0);
    check("the slow rare site (60 ms in 3 holds) is listed before the frequent one (50 ms in 500)", text.find("slow_rare:20") < text.find("frequent:10"));
    check("hold counts are shown", text.find("slow_rare:20 3,") != std::string::npos && text.find("frequent:10 500,") != std::string::npos);
    check("hold total and max are in ms", text.find("60/20.0") != std::string::npos);
    check("the wait of a site is shown", text.find(", 15]") != std::string::npos);
    check("two lines of one function are two sites", text.find("frequent:11 1,") != std::string::npos);
    check("the table is empty again after a line", !sites.top_line(line, sizeof(line), 1000.0));
    if (failures) return 1;
    std::printf("startup_lock_sites: PASS\n");
    return 0;
}
