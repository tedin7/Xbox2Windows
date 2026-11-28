Direct C++ compute lowers signed and unsigned 64-bit integer VOPC comparisons to DXIL i64 predicates with the existing VCC, SGPR, and EXEC lane-mask writes.
