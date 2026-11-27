"""Compile a captured graphics request as a matched desktop SM6 pair."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import gfx2hlsl as gfx


def bindings(lines, sampler_tags):
    resources = []
    samplers = []
    for line in lines:
        words = line.split()
        if words[:1] == ['srv']:
            resources.append(words)
        if words and words[0] in sampler_tags:
            samplers.append(tuple(int(word, 0) for word in words[1:]))
    return resources, samplers


def write_program(path, container):
    raw = gfx.strip_header(container)
    instructions = gfx.decode(raw)
    if not instructions or instructions[-1][2] != 's_endpgm':
        raise ValueError('missing program boundary')
    end, size, _, _ = instructions[-1]
    path.write_bytes(raw[:end + size])
    return len(container) - len(raw)


def build(marker, dest, compiler, sdk, cache, buffered_vertex=False):
    match = re.fullmatch(r'req_([0-9a-f]{16})\.rtv', marker.name)
    if not match:
        raise ValueError('expected a req_<16 hex digits>.rtv marker')
    variant = int(match[1], 16)
    request = marker.read_bytes()
    parsed = re.fullmatch(rb'rtv1 ([0-9]{1,20}) ' + (b'10' if buffered_vertex else b'26 54'), request)
    if not parsed or gfx.fnv(request) != variant:
        raise ValueError('expected a matching ' + ('RGBA16_FLOAT' if buffered_vertex else 'R11G11B10_FLOAT / R16_FLOAT') + ' request')
    base = int(parsed[1])
    stem = f'req_{base:016x}'
    dest.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest) as temp:
        bundle = Path(temp) / 'bundle'
        inputs = bundle / 'request'
        inputs.mkdir(parents=True)
        (inputs / marker.name).write_bytes(request)
        pieces = []
        for suffix in ('.a', '.b', '.root', '.ia'):
            path = marker.with_name(stem + suffix)
            if suffix == '.ia' and not path.exists():
                continue
            data = path.read_bytes()
            pieces.append(data)
            (inputs / path.name).write_bytes(data)
        if gfx.pipeline_key(*pieces) != base:
            raise ValueError('graphics stage, root or input layout key mismatch')
        stages, root, formats, rtv, fetch, _, attr_map = gfx.load_request(str(inputs), variant)
        if any(blob.startswith(b'DXBC') for blob in pieces[:2]) or fetch is not None or attr_map is not None or len(pieces) != 3:
            raise ValueError('this profile requires the raw two-table graphics interface')
        tables = [{'type': 0, 'vis': 0}] * 2
        if gfx.parse_root(root) != {'flags': 1, 'params': tables}:
            raise ValueError('unexpected graphics root layout')
        pair = gfx.Pipeline(stages['vs'], stages['ps'], root, sampler_heap=False, rtv_formats=formats)
        manifest = pair.manifest()
        if pair.params != ([0, 1] if buffered_vertex else [0]) or pair.flat or pair.notes or pair.lod_unvalidated:
            raise ValueError('unsupported vertex parameters or unverified binding manifest')
        name = match[1]
        ps = bundle / (name + '.ps.bin')
        ps_offset = write_program(ps, stages['ps'])
        vs = bundle / (name + '.vs.bin')
        vs_offset = write_program(vs, stages['vs'])
        entry = ['udsrc 0 rect', 'udsrc 1 basevertex', 'udsrc 2 startinstance']
        if buffered_vertex:
            entry.append('udsrc 3 stride 0')
        if manifest.splitlines()[1:len(entry)+2] != [f'ud {len(entry)}', *entry]:
            raise ValueError('unexpected vertex entry constants')
        pixel_source = ps
        if buffered_vertex:
            pixel_source = bundle / (name + '.ps.hlsl')
            pixel_source.write_text(pair.hlsl_ps(), encoding='utf-8')
        batch = 'gfx1013-buffered-pair' if buffered_vertex else 'gfx1013-sampled-pair'
        result = subprocess.run([compiler, sdk, cache, '--jobs', '2', batch,
                                 vs, bundle / (name + '.vs.cso'), pixel_source, bundle / (name + '.ps.cso')],
                                check=True, text=True, stdout=subprocess.PIPE)
        print(result.stdout, end='')
        actual = bindings(result.stdout.splitlines(), {'sampler'})
        expected = bindings(manifest.splitlines(), {'smp', 'smpl'})
        if buffered_vertex:
            rows, _ = actual
            if not rows or rows != expected[0][:len(rows)]:
                raise ValueError('native vertex buffer registers disagree with paired graphics manifest')
        elif actual != expected:
            raise ValueError('native pixel bindings disagree with paired graphics manifest')
        (bundle / (name + '.txt')).write_text(manifest + rtv + '\n', encoding='utf-8')
        metadata = {'profile': 'gfx1013-rgb11-r16-ps', 'request': 'request/' + marker.name,
                    'pixel_offset': ps_offset, 'pixel_bytes': ps.stat().st_size,
                    'vertex_profile': 'gfx1013-vertex-id-vs', 'vertex_offset': vs_offset, 'vertex_bytes': vs.stat().st_size,
                    'entry': {'table_sgprs': [8, 9], 'primitive_mask_sgpr': 10,
                              'perspective_center_vgprs': [0, 1], 'attr0': 'PARAM0'},
                    'rtv_formats': list(formats),
                    'sha256': {file.relative_to(bundle).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                               for file in sorted(bundle.rglob('*')) if file.is_file()}}
        if buffered_vertex:
            metadata.update(profile='ps_6_0', vertex_profile='gfx1013-buffered-vs',
                            entry={'vertex_table_sgprs': [0, 1], 'base_vertex_sgpr': 2, 'root_table_sgprs': [8, 9],
                                   'vertex_stride_word': 3, 'pixel_frontend': 'gfx2hlsl'})
        record = json.dumps(metadata, indent=2, sort_keys=True).encode() + b'\n'
        (bundle / 'replay.json').write_bytes(record)
        output = dest / (name + '-' + hashlib.sha256(record).hexdigest())
        try:
            bundle.rename(output)
        except OSError:
            if not output.is_dir() or (output / 'replay.json').read_bytes() != record:
                raise
            for path, expected in metadata['sha256'].items():
                if hashlib.sha256((output / path).read_bytes()).hexdigest() != expected:
                    raise ValueError('damaged replay bundle: ' + path)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('marker', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--buffered-vertex', action='store_true', help='translate a raw buffered VS with an RGBA16_FLOAT pixel companion')
    for option in ('compiler', 'sdk', 'cache'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    print(build(args.marker, args.output, args.compiler, args.sdk, args.cache, args.buffered_vertex))
