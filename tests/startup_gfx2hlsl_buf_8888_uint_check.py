"""BUF_FMT_8_8_8_8_UINT in tools/startup/gfx2hlsl.py: a typed buffer load reads four unsigned bytes, unscaled.

Pure Python: the translated lines of a tbuffer load are evaluated against a word, no compiler needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402


class Recorder:
    def __init__(self):
        self.lines = []

    def emit(self, text):
        self.lines.append(text)

    def set_v(self, n, expr):
        self.lines.append((n, expr))


def translate():
    rec = Recorder()
    gfx2hlsl.Stage.load_format(rec, 'BUF_FMT_8_8_8_8_UINT', 't0', 8, 4)
    return rec.lines


def evaluate(word):
    """Run the emitted lines for one loaded word; returns the four component values."""
    lines = translate()
    load = lines[0]
    assert load.startswith('w = ldu(t0, a)'), load
    env = {'w': word}
    return [eval(expr.replace('U(', '(').replace('u', ''), {}, env) for _, expr in lines[1:]]


def main():
    for word in (0x00000000, 0xffffffff, 0x80402010, 0x01fe7f80):
        want = [word & 255, (word >> 8) & 255, (word >> 16) & 255, word >> 24]
        got = evaluate(word)
        assert got == want, (hex(word), got, want)
    assert [n for n, _ in translate()[1:]] == [8, 9, 10, 11]
    # The vertex-fetch twin table: DXGI R8G8B8A8_UINT (30) names this format, four bytes wide.
    assert gfx2hlsl.FETCH_DXGI[30] == ('BUF_FMT_8_8_8_8_UINT', 4), gfx2hlsl.FETCH_DXGI.get(30)
    assert all(name in gfx2hlsl.FETCH_FORMATS for name, _ in gfx2hlsl.FETCH_DXGI.values())
    print('ok')


if __name__ == '__main__':
    main()
