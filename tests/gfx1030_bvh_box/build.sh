#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu

cd "$(dirname "$0")"
output=../../out/native-gfx1030-bvh-box
mkdir -p "$output"
"${LLVM_MC_15:-llvm-mc-15}" -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj box.s -o "$output/box.o"
"${LD_LLD_15:-ld.lld-15}" -shared "$output/box.o" -o "$output/box.hsaco"
"${LLVM_OBJDUMP_15:-llvm-objdump-15}" -d --mcpu=gfx1030 "$output/box.hsaco" > "$output/box.disasm"
for op in image_bvh_intersect_ray image_bvh64_intersect_ray; do
    test "$(grep -c "[[:space:]]$op[[:space:]]" "$output/box.disasm")" -eq 1
    grep "[[:space:]]$op[[:space:]]" "$output/box.disasm"
done
sha256sum "$output/box.hsaco"
