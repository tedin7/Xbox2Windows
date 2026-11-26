# Graphics scalar loads through joined table pointers

At a forward join, each incoming edge selects its table and byte offset. The
translator requires a compatible pointer pair on every edge and supports up to
four distinct addresses. The graphics runtime snapshots bounded table words
into the translated draw's entry constants and binds every candidate texture
view. The shader selects numeric words from the entry constants and switches
between the texture bindings. Five addresses, an untracked advance, a
loop-carried advance, a mismatched high word, and a non-pointer arm retain
their previous refusals.

The source-built fixture draws one strip per entry value with two, three, and
four tables. Each table supplies a distinct R8 texture and a distinct scalar
word. The test compares all 32 RGBA pixels with a CPU reference. It recompiles
the pixel shader with the selector forced to zero and requires the readback to
match table zero in every strip and differ from the original reference.

On Linux, the host control tests the accepted bindings and retained refusals.
The optional binary check compares the encoded fixture words with assembly
for gfx1013.

```sh
python3 -B tests/startup_joined_table_pointer_check.py
python3 -B tests/startup_joined_table_pointer_check.py binary llvm-mc-15
```

Run the draw and mutant on a Windows build with WARP and the shader compilers.

```powershell
ctest --preset dev --no-tests=error -R '^startup_joined_table_pointer_(cpu|choice_(fxc|dxc)_warp)$'
```
