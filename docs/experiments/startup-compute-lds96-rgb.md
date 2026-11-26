# Startup compute 96-bit LDS and RGB image store

The 96-bit LDS fixture uses 64 threads, a 12-byte lane stride and a four-byte
instruction offset. After a group barrier, each half reads a record from the
opposite half. The read aliases its address register with its first destination.
Each of the two controls substitutes the second dword for the third, on the
store or the read. Each must produce 64 mismatched words out of 320. The
translator applies the existing straight-line LDS exchange proof, which requires
bounded fills and reads, wait counts and a uniform group barrier.

The RGB fixture stores three distinct values at mip zero while EXEC disables the
second row. The CPU reference compares 1,344 words across all three mips,
including alpha and the untouched row. A control replaces the UAV load with
zero and must produce eight alpha mismatches. The read-modify-write requires
typed UAV loads for the bound format and assumes that no concurrent write
changes the preserved channel at the same texel.

Run the offline and WARP checks on Windows from the configured dev preset.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_(lds_b128_(controls|warp|t-wrong-store_warp|t-wrong-read_warp)|image_store(_rgb_warp)?)$'
```

Run the synthetic checks on a host with Python and a gfx1013-capable assembler.

```sh
python3 -B tests/startup_compute_lds_b128_check.py
python3 -B tests/startup_compute_lds_b128_check.py binary llvm-mc-19
python3 -B tests/startup_compute_image_store_check.py --llvm-mc llvm-mc-19
```

On Linux x86-64 with Python 3.14.8, the three commands passed. The two new
programs reproduced their original refusals on baseline `e8015da7a`. Eleven
existing compute fixtures produced identical HLSL and manifests against that
baseline. GPU readback is pending on Windows.
