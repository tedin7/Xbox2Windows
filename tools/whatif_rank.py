#!/usr/bin/env python3
"""What-if ranking, run ON THE POD (stdlib only).

For the top N refused graphics keys (by use count in a base pass2 log) it stands the key's refused pixel shader in by a flat
mid-grey (X2W_REPLAY_WHATIF=<key>:const), replays the recording once (pass2 only) and scores how much the final presented
picture changes against the base run: share of non-black pixels and mean level. Output: whatif_rank.tsv with
key, stage, mode, nonblack_before, nonblack_after, delta (nonblack share), mean_before, mean_after.

usage: whatif_rank.py <cache dir> <base pass2 dir (stderr.txt + present_*.png)> <out dir> [--top N] [--recording FILE]
       [--player DIR] [--keys k1,k2,...]
The cache dir is copied per key (the original stays untouched); the translator is the one under --player/../tools or /root/replay/tools.
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib


def png_stats(path):
    data = open(path, 'rb').read()
    at, idat, size, color, depth = 8, b'', None, None, None
    while at < len(data):
        length, tag = struct.unpack_from('>I4s', data, at)
        body = data[at + 8:at + 8 + length]
        if tag == b'IHDR':
            size = struct.unpack_from('>II', body)
            depth, color = body[8], body[9]
        elif tag == b'IDAT':
            idat += body
        at += 12 + length
    width, height = size
    channels = {2: 3, 6: 4, 0: 1, 4: 2}[color]
    stride = width * channels
    raw = zlib.decompress(idat)
    rows, prev = [], bytearray(stride)
    pos = 0
    for _ in range(height):
        kind = raw[pos]
        line = bytearray(raw[pos + 1:pos + 1 + stride])
        pos += 1 + stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + a) & 255
            elif kind == 2:
                line[i] = (line[i] + b) & 255
            elif kind == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    nonblack = total = 0
    level = 0
    step = 4  # every fourth pixel in both directions: enough for a share and a mean
    for y in range(0, height, step):
        line = rows[y]
        for x in range(0, width, step):
            r, g, b = line[x * channels], line[x * channels + 1], line[x * channels + 2]
            total += 1
            level += r + g + b
            if r or g or b:
                nonblack += 1
    return nonblack / total, level / (3.0 * total)


def last_picture(folder):
    names = sorted(n for n in os.listdir(folder) if re.fullmatch(r'present_\d+\.png', n))
    return os.path.join(folder, names[-1]) if names else None


def ranked_keys(base_dir, cache, top):
    uses = {}
    for line in open(os.path.join(base_dir, 'stderr.txt'), errors='replace'):
        m = re.search(r'x2w-render key use graphics ([0-9a-f]{16}) .* uses (\d+)', line)
        if m:
            uses[m.group(1)] = max(uses.get(m.group(1), 0), int(m.group(2)))
        m = re.search(r'x2w-render not drawn key ([0-9a-f]{16}): (\d+)', line)
        if m:
            uses[m.group(1)] = max(uses.get(m.group(1), 0), int(m.group(2)))
    refused = [k for k in uses if os.path.exists(os.path.join(cache, k + '.fail'))]
    refused.sort(key=lambda k: -uses[k])
    return [(k, uses[k]) for k in refused[:top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache')
    ap.add_argument('base')
    ap.add_argument('out')
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--recording', default='/root/replay/rec/replay.x2wr')
    ap.add_argument('--player', default='/root/replay/player')
    ap.add_argument('--keys', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base_picture = last_picture(args.base)
    before_share, before_level = png_stats(base_picture)
    keys = [(k, 0) for k in args.keys.split(',') if k] or ranked_keys(args.base, args.cache, args.top)
    rows = []
    env = dict(os.environ, DISPLAY=':99', WINEPREFIX='/root/pfx', WINEDEBUG='-all', XDG_RUNTIME_DIR='/tmp',
               WINEDLLOVERRIDES='d3d12,d3d12core,dxgi=n', WINEFSYNC='1', DXVK_LOG_LEVEL='none', VKD3D_DEBUG='none',
               X2W_DXC='/root/dxcwrap.sh', X2W_FXC='/root/fxcwrap.sh')
    for key, used in keys:
        work = os.path.join(args.out, 'cache_' + key)
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(args.cache, work)
        for suffix in ('.fail', '.txt'):
            try:
                os.remove(os.path.join(work, key + suffix))
            except FileNotFoundError:
                pass
        run_env = dict(env, X2W_REPLAY_WHATIF=key + ':const')
        subprocess.run(['python3', 'tools/startup/gfx2hlsl.py', '--pending', work], cwd='/root/replay', env=run_env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        published = os.path.exists(os.path.join(work, key + '.txt'))
        stage = 'ps' if published else 'unpublished'
        pictures = os.path.join(args.out, 'pass2_' + key)
        shutil.rmtree(pictures, ignore_errors=True)
        os.makedirs(pictures)
        after_share, after_level = before_share, before_level
        if published:
            player_env = dict(run_env, X2W_SHADER_CACHE='Z:' + work.replace('/', '\\'), X2W_DEBUG_PRESENT_TARGET='',
                              X2W_DEBUG_PRESENT_EVERY='')
            for name in ('X2W_DEBUG_PRESENT_TARGET', 'X2W_DEBUG_PRESENT_EVERY'):
                player_env.pop(name, None)
            with open(os.path.join(pictures, 'stdout.txt'), 'w') as out, open(os.path.join(pictures, 'stderr.txt'), 'w') as err:
                subprocess.run(['wine', 'x2w_replay.exe', 'Z:' + args.recording.replace('/', '\\'),
                                'Z:' + pictures.replace('/', '\\'), '--adapter', 'hardware'],
                               cwd=args.player, env=player_env, stdout=out, stderr=err)
            picture = last_picture(pictures)
            if picture:
                after_share, after_level = png_stats(picture)
        rows.append((key, stage, 'const', before_share, after_share, after_share - before_share, before_level, after_level, used))
        shutil.rmtree(work, ignore_errors=True)
        with open(os.path.join(args.out, 'whatif_rank.tsv'), 'w') as tsv:
            tsv.write('key\tstage\tmode\tnonblack_before\tnonblack_after\tdelta\tmean_before\tmean_after\tuses\n')
            for r in sorted(rows, key=lambda r: -r[5]):
                tsv.write('%s\t%s\t%s\t%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%d\n' % r)
        print('whatif %s %s uses %d: nonblack %.4f -> %.4f' % (key, stage, used, before_share, after_share), flush=True)


if __name__ == '__main__':
    sys.exit(main())
