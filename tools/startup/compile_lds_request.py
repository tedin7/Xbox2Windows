"""Replay a forward-only compute capture with typed image feedback and workgroup LDS."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory

import gfx2hlsl as gfx


class ComputeStage(gfx.Stage):
    def __init__(self, pipe, kind, code, **options):
        self.code = gfx.strip_header(code)
        super().__init__(pipe, kind, code, **options)

    def lower(self, op, asm, args, mods):
        if op in {'v_add_f32', 'v_sub_f32', 'v_subrev_f32', 'v_mul_f32',
                  'v_mul_u32_u24', 'v_lshlrev_b32', 'v_add_nc_u32'} and asm.startswith(op+'_e32 ') and not mods:
            word, = struct.unpack_from('<I', self.code, self.pc)
            src0 = word & 511
            if src0 >= 255 or 128 <= src0 <= 208:
                # Leave the frontend's operand and register-liveness checks in place.
                self.u(args[1])
                self.u(args[2])
                self.record_v_write(int(args[0][1:]))
                raw = f'{word:08x}'
                if src0 == 255:
                    literal, = struct.unpack_from('<I', self.code, self.pc+4)
                    raw += f' {literal:08x}'
                self.emit('#pragma x2w_vop2 '+raw)
                return
        if op.startswith('ds_'):
            if not 0 < self.pipe.lds_words <= 8192:
                raise gfx.Unsupported('native LDS requires a nonzero allocation of at most 32 KiB')
            self.uses_lds = True
            has_result = op.startswith('ds_read') or '_rtn_' in op
            for operand in args[1:] if has_result else args:
                reg = gfx.reg_range(operand)
                if not reg or reg[0] != 'v':
                    raise gfx.Unsupported('expected local DS VGPR operands')
                self.u(operand)
            if has_result:
                # Check all inputs first: the destination can alias the address or data.
                _, lo, hi = gfx.reg_range(args[0])
                for index in range(lo, hi+1):
                    self.record_v_write(index)
            first, second = struct.unpack_from('<II', self.code, self.pc)
            self.emit(f'#pragma x2w_lds {first:08x} {second:08x}')
            return
        if op == 's_barrier' and not mods and args in ([], ['0'], ['0x0']):
            self.emit('GroupMemoryBarrierWithGroupSync();')
            return
        if op == 's_waitcnt_depctr' and args == ['0xffe3'] and not mods:
            return  # register dependencies are explicit in the generated HLSL
        if op in ('image_load', 'image_store') and self.fixed_table_image(args[2],op) == self.pipe.feedback:
            writing = op == 'image_store'
            required = {'dmask:0xf' if writing else 'dmask:0x7', 'dim:SQ_RSRC_IMG_2D', 'unorm'}
            packed = 'a16' in mods
            if set(mods) != (required | ({'a16'} if packed else set())):
                raise gfx.Unsupported('unsupported typed image feedback instruction')
            _, x, y = gfx.reg_range(args[1])
            _, lo, hi = gfx.reg_range(args[0])
            coordinate_registers = 2 if writing or not packed else 1
            if hi-lo != (3 if writing else 2) or y-x+1 != coordinate_registers:
                raise gfx.Unsupported('unexpected image feedback register count')
            address = self.u(f'v{x}')
            coord = f'uint2({address}&65535u,{address}>>16)' if packed else f'uint2({address},{self.u(f"v{y}")})'
            uav = self.pipe.uav(self.pipe.feedback, 'tex2d')
            if writing:
                values = ','.join(self.u(f'v{i}') for i in range(lo,hi+1))
                self.emit(f'if(exec&1u) {uav}[{coord}]=asfloat(uint4({values}));')
            else:
                self.emit(f't={uav}[{coord}];')
                for i in range(3):
                    self.set_v(lo+i,'asuint(t.'+'xyz'[i]+')')
            return
        return super().lower(op,asm,args,mods)

    def body(self):
        # A barrier can be emitted outside per-wave control only when all paths
        # from entry pass through its block. Reject loops in this frontend.
        index = {offset:i for i,(offset,_,_) in enumerate(self.blocks)}
        successors = []
        for i,(_,_,term) in enumerate(self.blocks):
            if not term:
                dest = [i+1]
            elif term[0] == 'end':
                dest = []
            elif term[0] == 'jump':
                dest = [index[term[1]]]
            elif term[0] == 'branch':
                dest = [i+1,index[term[2]]]
            else:
                raise gfx.Unsupported('unsupported compute branch')
            if any(not i < d < len(index) for d in dest):
                raise gfx.Unsupported('native LDS requires forward control flow')
            successors.append(dest)
        post = {}
        for i in reversed(range(len(index))):
            dest = successors[i]
            post[i] = {i} | (set.intersection(*(post[j] for j in dest)) if dest else set())

        lines = ['uint pc=0;']
        for i,(_,statements,term) in enumerate(self.blocks):
            guard = f'if(pc=={i}u){{'
            lines.append(guard)
            for statement in statements:
                if statement == 'GroupMemoryBarrierWithGroupSync();':
                    if i not in post[0]:
                        raise gfx.Unsupported('workgroup barrier is conditional')
                    lines.extend(['}',statement,guard])
                else:
                    lines.append(statement)
            dest = successors[i]
            if len(dest) == 2:
                cond = term[1]
                for mask in ('exec','vcc'):
                    for op,neg in [('==','!'),('!=','')]:
                        if cond == f'({mask} & 1) {op} 0':
                            cond = f'{neg}WaveActiveAnyTrue(({mask}&1u)!=0u)'
                lines.append(f'pc=({cond})?{dest[1]}u:{dest[0]}u;')
            else:
                lines.append(f'pc={dest[0] if dest else len(index)}u;')
            lines.append('}')
        return lines


class ComputePipeline(gfx.ComputePipeline):
    stage_type = ComputeStage

    def __init__(self, *args, feedback, **kwargs):
        self.feedback = feedback
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('marker',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--compiler',required=True,type=Path)
    parser.add_argument('--sdk',required=True,type=Path)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--rw-image',required=True,nargs=2,type=int,metavar=('PARAM','BYTE_OFFSET'))
    parser.add_argument('--wave-size',required=True,type=int,choices=[32,64])
    args = parser.parse_args()
    name = args.marker.name
    if not re.fullmatch(r'req_[0-9a-f]{16}\.c67',name):
        parser.error('expected req_<16 hex digits>.c67')
    key = int(name[4:20],16)
    param,off = args.rw_image
    if param < 0 or off < 0 or off%32:
        parser.error('expected a nonnegative parameter and a 32-byte-aligned descriptor offset')
    args.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output) as temp:
        bundle = Path(temp)/'bundle'
        request = bundle/'request'
        request.mkdir(parents=True)
        for ext in ('.c67','.root','.data','.desc','.sa','.sb'):
            path = args.marker.with_suffix(ext)
            shutil.copyfile(path,request/path.name)
        pipeline = gfx.load_compute_request(str(request),key,
            pipeline_type=partial(ComputePipeline,feedback=(('table',param),off)))
        source = pipeline.hlsl_cs().replace('[numthreads(',f'[WaveSize({args.wave_size})]\n[numthreads(')
        stage = bundle/f'{key:016x}.cs.hlsl'
        stage.write_text(source,encoding='utf-8')
        bytecode = stage.with_suffix('.cso')
        subprocess.run([args.compiler,args.sdk,args.cache,'cs_6_6',stage,bytecode],check=True)
        manifest = bundle/f'{key:016x}.txt'
        manifest.write_text(pipeline.manifest(),encoding='utf-8')
        metadata = {'schema':1,'request':f'request/{name}','profile':'cs_6_6','rw_image':[param,off],
                    'wave_size':args.wave_size,'sha256':{}}
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                metadata['sha256'][path.relative_to(bundle).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        replay = json.dumps(metadata,indent=2,sort_keys=True)+'\n'
        (bundle/'replay.json').write_bytes(replay.encode())
        dest = args.output/f'{key:016x}-{hashlib.sha256(replay.encode()).hexdigest()}'
        try:
            bundle.rename(dest)
        except OSError:
            if not dest.is_dir() or (dest/'replay.json').read_bytes()!=replay.encode():
                raise
            for relative,digest in metadata['sha256'].items():
                if hashlib.sha256((dest/relative).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError(f'damaged replay artifact {relative}')
        print(dest)


if __name__ == '__main__':
    main()
