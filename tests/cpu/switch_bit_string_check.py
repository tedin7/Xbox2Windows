"""Compare signed register-indexed memory bit tests with native x64 and a byte model."""
import ctypes
from pathlib import Path
import sys

from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from cpu_switch_aot import Lifter, translate

MASK = (1 << 64) - 1
OPS = ('bt', 'bts', 'btr', 'btc')


def expected(data, base, bits, offset, op):
    result = bytearray(data)
    size = bits // 8
    address = base + (offset // bits) * size
    value = int.from_bytes(result[address:address + size], 'little')
    bit = offset % bits
    carry = (value >> bit) & 1
    if op == 'bts':
        value |= 1 << bit
    elif op == 'btr':
        value &= ~(1 << bit)
    elif op == 'btc':
        value ^= 1 << bit
    result[address:address + size] = value.to_bytes(size, 'little')
    return carry, bytes(result)


def run(fn, data, base, offset):
    array = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    fn.argtypes = (ctypes.c_uint64, ctypes.c_void_p)
    fn.restype = ctypes.c_uint64
    carry = fn(offset & MASK, ctypes.addressof(array) + base)
    return carry, bytes(array)


def check_refusals(original_path):
    for code, frame in (('0f a3 08', True), ('67 0f a3 08', False)):
        ins = next(Decoder(64, bytes.fromhex(code), ip=0x1000))
        lifter = Lifter()
        if frame:
            lifter.frame[ins.ip] = 0
        try:
            lifter.lower(ins)
        except ValueError as error:
            if 'unsupported bit-string operand' not in str(error):
                raise
        else:
            raise AssertionError(f'accepted unsupported bit-string {code}')
    for name, reason, image_data in (
            ('neg_bit_frame', 'unsupported bit-string operand', False),
            ('neg_bit_image', 'unsupported bit-string operand', True),
            ('neg_bit_undef', 'undefined register r10', False)):
        try:
            translate(original_path, [name], image_data=image_data)
        except ValueError as error:
            if reason not in str(error):
                raise AssertionError(f'{name}: expected {reason!r}, got {error}') from error
        else:
            raise AssertionError(f'accepted unsupported {name}')


def check_faults(libraries):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
    kernel.VirtualAlloc.restype = ctypes.c_void_p
    kernel.VirtualProtect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_uint32))
    kernel.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    page = 4096
    address = kernel.VirtualAlloc(None, page * 2, 0x3000, 0x04)
    if not address:
        raise ctypes.WinError(ctypes.get_last_error())
    old = ctypes.c_uint32()
    try:
        for protection, names, offset, base in (
                (0x01, ('bt64_dead',), -64, address + page),
                (0x02, ('bts64_string', 'btr64_string', 'btc64_string'), 0, address)):
            for fill in (0, 0xff):
                if not kernel.VirtualProtect(address, page, 0x04, ctypes.byref(old)):
                    raise ctypes.WinError(ctypes.get_last_error())
                ctypes.memset(address, fill, page)
                if not kernel.VirtualProtect(address, page, protection, ctypes.byref(old)):
                    raise ctypes.WinError(ctypes.get_last_error())
                for library, prefix in libraries:
                    for name in names:
                        fn = getattr(library, prefix + name)
                        fn.argtypes = (ctypes.c_uint64, ctypes.c_void_p)
                        fn.restype = ctypes.c_uint64
                        try:
                            fn(offset & MASK, base)
                        except OSError as error:
                            if 'access violation' not in str(error).lower():
                                raise
                        else:
                            raise AssertionError(f'{prefix}{name} did not fault with protection {protection:#x}')
    finally:
        if not kernel.VirtualFree(address, 0, 0x8000):
            raise ctypes.WinError(ctypes.get_last_error())


def check(native_path, lifted_path):
    native, lifted = ctypes.CDLL(native_path), ctypes.CDLL(lifted_path)
    comparisons = 0
    masked_faults = 0
    for bits in (16, 32, 64):
        offsets = (-bits - 1, -bits, -bits + 1, -1, 0, 1, bits - 1, bits, bits + 1)
        for op in OPS:
            name = f'{op}{bits}_string'
            source = getattr(native, name)
            translation = getattr(lifted, 'lifted_' + name)
            for base in (29, 32):
                for pattern in (0, 1, 2):
                    data = bytes(((i * 47 + pattern * 83) ^ (i >> 1)) & 255 for i in range(64))
                    for offset in offsets:
                        # High bits outside the operand width must not affect the signed offset.
                        encoded = offset & ((1 << bits) - 1)
                        if bits < 64:
                            encoded |= (0xa5a5a5a5a5a5a5a5 & ~((1 << bits) - 1))
                        want = expected(data, base, bits, offset, op)
                        original = run(source, data, base, encoded)
                        actual = run(translation, data, base, encoded)
                        if original != want or actual != want:
                            raise AssertionError(f'{name}: {base=} {offset=} {original=} {actual=} {want=}')
                        masked_faults += want != expected(data, base, bits, offset % bits, op)
                        comparisons += 1
    for offset in (-9, -1, 0, 1, 9):
        data = bytes((i * 47 + 83) & 255 for i in range(64))
        want = expected(data, 32 + offset, 64, offset, 'btc')
        for library, prefix in ((native, ''), (lifted, 'lifted_')):
            actual = run(getattr(library, prefix + 'btc64_alias'), data, 32, offset)
            if actual != want:
                raise AssertionError(f'{prefix}btc64_alias: {offset=} {actual=} {want=}')
    if not masked_faults:
        raise AssertionError('masking the offset did not change any expected result')
    check_refusals(native_path)
    check_faults(((native, ''), (lifted, 'lifted_')))
    print(f'{comparisons} native, translated and reference comparisons passed; '
          f'{masked_faults} cases distinguish a masked-offset fault; alias, fault and refusal controls passed')


if __name__ == '__main__':
    check(*sys.argv[1:])
