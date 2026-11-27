"""Serial wave64 lowering for slot 57 mesh shaders."""

from copy import deepcopy
import os
import re


BRANCHES = {
    's_cbranch_scc0': '!scc', 's_cbranch_scc1': 'scc',
    's_cbranch_execz': '(exec_lo | exec_hi) == 0u',
    's_cbranch_execnz': '(exec_lo | exec_hi) != 0u',
    's_cbranch_vccz': '(s106 | s107) == 0u',
    's_cbranch_vccnz': '(s106 | s107) != 0u',
}
NO_EFFECT = ('s_waitcnt', 's_nop', 's_inst_prefetch', 's_clause')
CROSS_LANE = ('v_readfirstlane_b32', 'v_readlane_b32', 'v_writelane_b32')
MASK_WORDS = {'exec_lo', 'exec_hi', 'vcc_lo', 'vcc_hi'}
VCC_REGISTERS = {'vcc': 's106', 'vcc_lo': 's106', 'vcc_hi': 's107'}
UNDEFINED_ENTRY = ('an entry value the mesh model does not define',)
JOINED = ('a value joined across wave-loop blocks',)
LANE = '((wb + l) & 127u)'
LEAK = re.compile(r'\b(?:SV_GroupIndex|Wave\w*|GroupMemoryBarrier\w*|IsHelperLane|'
                  r'QuadRead\w*|dd[xy]\w*|tid|exec|vcc|uint64_t|mesh_(?:first|vote|ballot)\w*)\b')


def proof_family(G, reason):
    if reason.startswith('slot 57 mesh: export target '):
        return False
    return (reason.startswith(('slot 57 mesh', 'wave-wide read in a program with a branch'))
            or reason == G.READFIRSTLANE_UNIFORM
            or bool(re.search(r'reads lane mask .* as numeric data|numeric use of s\d+ holding '
                              r'(?:joined at a branch target|the saved mesh lane-0 mask)', reason)))


def is_branch(name):
    return name == 's_branch' or name.startswith('s_cbranch_')


def is_native(name, ops):
    scalar_mask = name.startswith('s_') and (
        '64' in name or 'saveexec' in name or 'wrexec' in name or any(x in MASK_WORDS | {'exec', 'vcc'} for x in ops))
    return (scalar_mask or is_branch(name)
            or name.startswith(NO_EFFECT + CROSS_LANE + ('ds_', 'v_mbcnt_', 'v_permlane',
                                                         's_getpc', 's_setpc', 's_swappc'))
            or '_dpp' in name
            or name in ('s_endpgm', 's_barrier', 's_sendmsg')
            or (name == 'exp' and ops[0].startswith('target_20 ')))


def registers(G, operands, kind):
    result = set()
    for operand in operands:
        for token in re.findall(r'\b[sv](?:\[\d+:\d+\]|\d+\b)', operand):
            bank, first, last = G.reg_range(token)
            if bank == kind:
                result.update('%s%d' % (kind, n) for n in range(first, last + 1))
    return result


def mask_operands(G, name, ops):
    """Operands that represent one bit per lane in the shared VALU lowering."""
    base = re.sub(r'_e(?:32|64)$', '', name)
    masks = []
    if base.startswith('v_cmp') and not base.startswith('v_cmpx') and len(ops) == 3:
        masks.append(ops[0])
    if base.startswith(G.TWO_DESTINATIONS) and len(ops) >= 2:
        masks.append(ops[1])
    if (base.startswith('v_cndmask') and len(ops) == 4) or '_co_ci_' in base:
        masks.extend(ops[-1:])
    return sorted({r[1] for t in masks if (r := G.reg_range(t)) and r[0] == 's'})


def effects(G, name, asm):
    reads, writes, _ = G.scalar_effects(name, asm)
    _, ops, _ = G.parse(asm)
    base = re.sub(r'_e(?:32|64)$', '', name)
    if name == 's_sendmsg':
        reads.add('m0')
    if (name in G.SCALAR_BITS and G.SCALAR_BITS[name][2]) or name in G.SCC_UNMODELLED or name.startswith(
            ('s_addc_', 's_subb_', 's_addk_', 's_lshl_', 's_lshr_', 's_ashr_', 's_bfe_', 's_wqm_', 's_quadmask_')):
        writes.add('scc')
    if name.startswith('v_cmpx'):
        # VOP3 encodes an unused scalar destination field.
        if name.endswith('_e64') and len(ops) == 3:
            reads = G.scalar_tokens(' '.join(ops[1:])) | {'exec'}
            ops = ops[1:]
        writes |= G.EXEC_KEYS
    if name.startswith(('v_cmp', 'v_add_co', 'v_sub_co', 'v_subrev_co')):
        if name.startswith('v_cmp') and not name.startswith('v_cmpx') and len(ops) == 2:
            reads = G.scalar_tokens(' '.join(ops))
            writes.add('vcc')
        if 'vcc' in writes:
            writes.add('vcc_hi')
        if 'exec' in writes:
            writes.add('exec_hi')
    if base.startswith('v_cndmask') and len(ops) == 3:
        reads.add('vcc')
    if name.startswith('s_cbranch_'):
        if 'vcc' in reads:
            reads.add('vcc_hi')
        if 'exec' in reads:
            reads.add('exec_hi')
    if 'saveexec' in name or 'wrexec' in name:
        reads |= G.EXEC_KEYS
        writes |= G.EXEC_KEYS
    for n in mask_operands(G, name, ops):
        if 's%d' % n in reads:
            reads.add('s%d' % (n + 1))
        if 's%d' % n in writes:
            writes.add('s%d' % (n + 1))
    if name.startswith(('v_cndmask', 'v_add_co_ci', 'v_sub_co_ci', 'v_subrev_co_ci')) and 'vcc' in reads:
        reads.add('vcc_hi')
    if name.startswith(('exp', 'ds_write', 'ds_permute', 'ds_bpermute', 'ds_swizzle')) or '_store' in name:
        dests, srcs = [], ops
        if name == 'exp':
            srcs = G.export_operands(asm)[1]
    elif name.startswith('v_cmp'):
        dests, srcs = [], ops
    elif name.startswith(('v_', 'buffer_', 'tbuffer_', 'image_', 'ds_read')):
        dests = G.destinations(base, ops)
        srcs = ops[len(dests):]
    else:
        dests, srcs = [], []
    if base.startswith(G.READ_MODIFY_WRITE) or base == 'v_swap_b32':
        srcs = srcs + dests
    return ({VCC_REGISTERS.get(reg, reg) for reg in reads} | registers(G, srcs, 'v'),
            {VCC_REGISTERS.get(reg, reg) for reg in writes} | registers(G, dests, 'v'))


class ControlFlow:
    def __init__(self, G, instructions):
        offsets = {off: i for i, (off, _, _, _) in enumerate(instructions)}
        leaders = {0}
        targets = {}
        for i, (_, _, name, asm) in enumerate(instructions):
            if is_branch(name):
                target = re.fullmatch(r'label_(0x[0-9a-f]+)', G.parse(asm)[1][0])
                if not target or int(target[1], 16) not in offsets:
                    raise G.Unsupported('wave loop: invalid branch target in ' + asm)
                targets[i] = offsets[int(target[1], 16)]
                leaders.add(targets[i])
            if (is_branch(name) or name in ('s_endpgm', 's_barrier')) and i + 1 < len(instructions):
                leaders.add(i + 1)
        starts = sorted(leaders)
        self.blocks = dict(zip(starts, starts[1:] + [len(instructions)]))
        self.successors = {}
        for start, end in self.blocks.items():
            name = instructions[end - 1][2]
            next_block = [end] if end in self.blocks else []
            self.successors[start] = ([targets[end - 1]] if name == 's_branch' else
                                      [targets[end - 1]] + next_block if is_branch(name) else
                                      [] if name == 's_endpgm' else next_block)
        self.targets = targets
        reachable = {0}
        todo = [0]
        while todo:
            for successor in self.successors[todo.pop()]:
                if successor not in reachable:
                    reachable.add(successor)
                    todo.append(successor)
        self.blocks = {start: end for start, end in self.blocks.items() if start in reachable}
        self.predecessors = {start: [] for start in self.blocks}
        for start in self.blocks:
            for successor in self.successors[start]:
                self.predecessors[successor].append(start)
        self.effects = [effects(G, name, asm) for _, _, name, asm in instructions]
        self.exec_zero_edges = {}
        for start, end in self.blocks.items():
            name = instructions[end - 1][2]
            if name not in ('s_cbranch_execz', 's_cbranch_execnz') or len(self.successors[start]) != 2:
                continue
            empty, active = self.successors[start]
            if name == 's_cbranch_execnz':
                empty, active = active, empty
            if empty == active:
                continue
            seen, todo, writes = {empty, start}, [active], set()
            while todo:
                block = todo.pop()
                if block in seen:
                    continue
                seen.add(block)
                for index in range(block, self.blocks[block]):
                    writes.update(reg for reg in self.effects[index][1]
                                  if re.fullmatch(r'[sv]\d+', reg))
                todo.extend(self.successors[block])
            self.exec_zero_edges[start, empty] = writes


    def definedness(self, entry, elided_scc=()):
        def transfer(state, index):
            if index in elided_scc:
                return (state | self.effects[index][1]) - {'scc'}
            return state | self.effects[index][1]

        top = entry | set().union(*(w for _, w in self.effects))
        exits = dict.fromkeys(self.blocks, top)
        inputs = {}
        while True:
            changed = False
            for start, end in self.blocks.items():
                incoming = [exits[p] | self.exec_zero_edges.get((p, start), set())
                            for p in self.predecessors[start]]
                if start == 0:
                    incoming.append(entry)
                inputs[start] = set.intersection(*incoming)
                result = inputs[start]
                for index in range(start, end):
                    result = transfer(result, index)
                if result != exits[start]:
                    exits[start] = result
                    changed = True
            if not changed:
                break
        before = {}
        for start, end in self.blocks.items():
            active = inputs[start]
            for index in range(start, end):
                before[index] = active
                active = transfer(active, index)
        return before


def meet(states, exec_zero=()):
    keys = set().union(*(state.keys() for state in states))
    result = {}
    for key in keys:
        values = [state.get(key) for state in states]
        first = values[0]
        active = [value for i, value in enumerate(values) if i not in exec_zero]
        if (active and active[0] is not None and all(value == active[0] for value in active)
                and all(value is None or value[0] != 'const' for value in values)
                and (active[0] != JOINED)):
            result[key] = active[0]
            continue
        if first is not None and all(value == first for value in values):
            result[key] = first
        elif any(value is not None and (value[0] != 'const' or len(value) != 2) for value in values):
            result[key] = JOINED
    return result


def translate_serial(G, ms, ps, root, pixel_refusal):
    class SerialMeshStage(G.Stage):
        serial_mesh = True

        def setup_entry(self):
            super().setup_entry()
            self.entry = ['s3 = wave_id << 24u;']
            self.entry_symbols = {k: v for k, v in self.sym.items() if v != UNDEFINED_ENTRY}
            entry = {'s3', 'exec', 'exec_hi', 'v5'}
            entry.update('s%d' % n for n in self.entry_symbols)
            self.defined_entry = entry

        def uniform_source(self, token):
            reg = G.reg_range(token)
            if reg and reg[0] == 's' and reg[1] == reg[2]:
                return True
            return token in MASK_WORDS | {'exec', 'vcc', 'm0'} or super().uniform_source(token)

        def translate(self):
            self.real = self.ins
            self.cfg = ControlFlow(G, self.real)
            self.before = self.cfg.definedness(self.defined_entry)
            pseudo = []
            self.index_at = {}
            for start, end in self.cfg.blocks.items():
                pseudo.append((self.real[start][0] * 2, 1, 'x2w_serial_block', 'x2w_serial_block %d' % start))
                for index in range(start, end):
                    off, size, name, asm = self.real[index]
                    self.index_at[off * 2 + 1] = index
                    if is_native(name, G.parse(asm)[1]):
                        name, asm = 'x2w_serial', 'x2w_serial %d' % index
                    pseudo.append((off * 2 + 1, size, name, asm))
            self.ins = pseudo
            prior = {}
            self.stub_mode = os.environ.get('X2W_MESH_WAVELOOP_STUBS') == '1'
            # A speculative loop pass must not leave resources or symbolic optimizations in the next pass.
            memo = {id(self.pipe): self.pipe, id(self.cfg): self.cfg, id(self.real): self.real, id(pseudo): pseudo}
            stage_entry = deepcopy(self.__dict__, memo.copy())
            pipeline_entry = deepcopy(self.pipe.__dict__)
            for pass_number in range(4):
                if pass_number:
                    self.__dict__.clear()
                    self.__dict__.update(deepcopy(stage_entry, memo.copy()))
                    self.pipe.__dict__.clear()
                    self.pipe.__dict__.update(deepcopy(pipeline_entry))
                self.elided_scc = set()
                self.previous_exits = prior
                self.exits, self.entries, self.errors = {}, {}, {}
                self.current_block = None
                super().translate()
                self.exits[self.current_block] = dict(self.sym)
                complete = {start: self.block_symbols(start, self.exits) for start in self.cfg.blocks}
                if complete == self.entries:
                    before = self.cfg.definedness(self.defined_entry, self.elided_scc)
                    for index, live in before.items():
                        if 'scc' in self.cfg.effects[index][0] and 'scc' not in live and index not in self.errors:
                            self.errors[index] = G.Unsupported('wave loop: scc may be read before it is written')
                    if self.errors and not self.stub_mode:
                        raise self.errors[min(self.errors)]
                    return
                prior = self.exits
            raise G.Unsupported('wave loop: symbolic block entries did not stabilize')

        def block_symbols(self, start, exits):
            predecessors = [p for p in self.cfg.predecessors[start] if p in exits]
            states = [exits[p] for p in predecessors]
            exec_zero = {i for i, p in enumerate(predecessors) if (p, start) in self.cfg.exec_zero_edges}
            if start == 0:
                states.append(self.entry_symbols)
            return meet(states, exec_zero) if states else {}

        def op_x2w_serial_block(self, ops, mods):
            if self.current_block is not None:
                self.exits[self.current_block] = dict(self.sym)
            start = self.current_block = int(ops[0])
            self.sym = self.block_symbols(start, self.previous_exits | self.exits)
            self.entries[start] = dict(self.sym)
            self.emit('//@B %d' % start)
            self.exec_generation += 1
            self.pinned_low16_v.update(self.low16_only_v)
            for field in ('align4_v', 'align4_s', 'uniform_v', 'uniform_expr', 'uniform_mask',
                          'float_const_s', 'heap_index', 'advanced_descriptor_words', 'uniform_cmp_masks'):
                getattr(self, field, {}).clear()
            self.last_v = None
            self.descriptor_block_first = self.descriptor_instruction_index + 1

        def read_vcc(self):
            # The current block's symbols, rather than Stage's linear VCC flags,
            # describe which values can reach this instruction.
            for n in (106, 107):
                symbol = self.sym.get(n)
                if symbol and not G.run_time_symbol(symbol):
                    raise G.Unsupported('wave loop: mask word s%d holds %s' % (n, symbol[0]))

        def descriptor(self, token, *, allow_null=True):
            # Keep Stage's extended-pair check even after a control-flow join.
            self.vcc_descriptor_extended = G.reg_range(token) == ('s', 100, 107)
            self.vcc_descriptor_overwritten = False
            return super().descriptor(token, allow_null=allow_null)

        def one(self, name, asm):
            if name == 'x2w_serial_block':
                return super().one(name, asm)
            index = self.index_at[self.pc]
            _, _, real_name, real_asm = self.real[index]
            _, ops, _ = G.parse(real_asm)
            self.emit('//@I %d' % index)
            first = len(self.lines)
            self.exec_known = self.exec_narrow_only = False
            self.exec_may_be_empty = True
            self.scc_verified = self.scc_uniform = 'scc' in self.before[index]
            self.masks = set(mask_operands(G, real_name, ops))
            self.mask_high.clear()
            getattr(self, 'uniform_cmp_masks', {}).clear()
            try:
                reads, writes = self.cfg.effects[index]
                missing = reads - self.before[index]
                if any(op in VCC_REGISTERS for op in ops) or real_name.startswith('s_cbranch_vcc'):
                    for reg in ('s106', 's107'):
                        symbol = self.sym.get(int(reg[1:]))
                        if reg in reads and symbol and not G.run_time_symbol(symbol):
                            raise G.Unsupported('wave loop: mask word %s holds %s' % (reg, symbol[0]))
                if name.startswith('s_load_') and len(ops) >= 2:
                    base = G.reg_range(ops[1])
                    symbol = self.sym.get(base[1]) if base and base[0] == 's' else None
                    if symbol and symbol[0] == 'ptr' and symbol[1][0] == 'table':
                        # Stage models this root table pointer as one word. Its SMEM pair spelling
                        # carries an unused high word in this descriptor contract.
                        missing.discard('s%d' % (base[1] + 1))
                missing_error = None
                if missing:
                    reg = min(missing)
                    missing_error = G.Unsupported('wave loop: %s may be read before it is written' % reg)
                    # In census mode retain the declaration of an export for pixel linkage.
                    if not (self.stub_mode and name == 'exp'):
                        raise missing_error
                for n in self.masks:
                    for word in (n, n + 1):
                        symbol = self.sym.get(word)
                        if 's%d' % word in reads and symbol and not G.run_time_symbol(symbol):
                            raise G.Unsupported('wave loop: mask word s%d holds %s' % (word, symbol[0]))
                if name == 'x2w_serial':
                    self.op_x2w_serial([str(index)], [])
                else:
                    if name == 'exp':
                        target, _, flags = G.export_operands(asm)
                        if not re.fullmatch(r'pos0|param\d+', target) or 'vm' in flags:
                            raise G.Unsupported('export target or form ' + asm)
                    super().one(name, self.numeric_mask_sources(name, asm))
                if missing_error:
                    raise missing_error
                if real_name.startswith('v_') and real_name not in CROSS_LANE:
                    for n in (106, 107):
                        if 's%d' % n in writes:
                            self.sym.pop(n, None)
                            self.align4_s.discard(n)
                for n in mask_operands(G, real_name, ops):
                    if 's%d' % (n + 1) in writes:
                        self.sym.pop(n + 1, None)
                        self.align4_s.discard(n + 1)
            except G.Unsupported as reason:
                self.errors[index] = reason
                del self.lines[first:]
                for reg in self.cfg.effects[index][1]:
                    if re.fullmatch(r's\d+', reg):
                        self.sym.pop(int(reg[1:]), None)
                self.emit('// X2W_SERIAL_STUB %s %s' % (real_name, ' '.join(str(reason).split())))
            finally:
                if 'scc' in self.cfg.effects[index][1] and self.scc_effect != 'emitted':
                    self.elided_scc.add(index)
                self.masks.clear()

        def numeric_mask_sources(self, name, asm):
            if not name.startswith('v_'):
                return asm
            _, ops, mods = G.parse(asm)
            base = re.sub(r'_e(?:32|64)$', '', name)
            if name.startswith('v_cmp'):
                first_source = int(len(ops) == 3)
            else:
                first_source = len(G.destinations(base, ops))
            last_source = len(ops)
            if (base.startswith('v_cndmask') and len(ops) == 4) or '_co_ci_' in base:
                last_source -= 1
            operands = list(ops)
            for i in range(first_source, last_source):
                operands[i] = re.sub(r'\b(?:exec|vcc)(?:_lo|_hi)?\b',
                                     lambda m: 'serial_word_' + m[0], ops[i])
                operands[i] = re.sub(r'\bs(\d+)\b',
                                     lambda m: 'serial_word_' + m[0] if int(m[1]) in self.masks else m[0],
                                     operands[i])
            if operands == ops:
                return asm
            return name + ' ' + ', '.join(operands) + (' ' + ' '.join(mods) if mods else '')

        def u(self, tok, *, low16_only=False):
            if tok.startswith('serial_word_'):
                token = tok.removeprefix('serial_word_')
                if token.startswith('s'):
                    value = super().S(int(token[1:]))
                    return 'serial_value_' + value if value == token else value
                return self.word(token)
            return super().u(tok, low16_only=low16_only)

        def S(self, n):
            if n in self.masks:
                self.used_s.add(n)
                return 's%d' % n
            return super().S(n)

        def word(self, token):
            if token in VCC_REGISTERS:
                return self.S(int(VCC_REGISTERS[token][1:]))
            if token in MASK_WORDS:
                return token
            if token in ('exec', 'vcc'):
                return token + '_lo'
            return self.u(token)

        def pair(self, token, destination=False):
            if token == 'exec':
                return 'exec_lo', 'exec_hi'
            if token == 'vcc':
                convert = self.destination if destination else self.word
                return convert('vcc_lo'), convert('vcc_hi')
            r = G.reg_range(token)
            if r and r[0] == 's' and r[2] == r[1] + 1:
                convert = self.destination if destination else self.word
                return tuple(convert('s%d' % n) for n in range(r[1], r[2] + 1))
            if not destination and not r and token not in MASK_WORDS | {'m0', 'src_scc'}:
                return self.word(token), '0xffffffffu' if re.fullmatch(r'-(?:[1-9]|1[0-6])', token) else '0u'
            raise G.Unsupported('wave loop: invalid scalar pair ' + token)

        def destination(self, token):
            token = VCC_REGISTERS.get(token, token)
            if token in MASK_WORDS:
                return token
            if token == 'null':
                self.used_scratch = True
                return 'discard_word'
            if token == 'm0':
                self.sym.pop('m0', None)
                return token
            r = G.reg_range(token)
            if not r or r[1] != r[2]:
                raise G.Unsupported('wave loop: invalid destination ' + token)
            if r[0] == 's':
                self.sym.pop(r[1], None)
                self.used_s.add(r[1])
            else:
                self.used_v.add(r[1])
                self.low16_only_v.discard(r[1])
                self.align4_v.discard(r[1])
            return token

        def write_pair(self, destination, expression, *, scc=False, save=False, exec_write=False):
            lo, hi = self.pair(destination, destination=True)
            self.emit('{ uint2 result64 = %s;' % expression)
            if save:
                self.emit('%s = exec_lo; %s = exec_hi;' % (lo, hi))
            else:
                self.emit('%s = result64.x; %s = result64.y;' % (lo, hi))
            if save or exec_write:
                self.emit('exec_lo = result64.x; exec_hi = result64.y;')
            if scc:
                self.set_scc('(result64.x | result64.y) != 0u', proof='0u')
            self.emit('}')

        def scalar_native(self, name, ops, mods):
            if mods:
                raise G.Unsupported('wave loop: modifiers on ' + name)
            if name in G.SCALAR_BITS or name == 's_ff1_i32_b64':
                pair, expr, scc = G.SCALAR_BITS.get(name, (True, G._ctz64, None))
                lo, hi = self.pair(ops[1]) if pair else (self.word(ops[1]), '0u')
                self.write_word(ops[0], expr(lo, hi), scc)
                return
            if name in ('s_cmp_eq_u64', 's_cmp_lg_u64'):
                args = [self.vector_pair(op) for op in ops]
                eq = 'all(%s == %s)' % tuple(args)
                self.set_scc('!' + eq if name == 's_cmp_lg_u64' else eq, proof='0u')
                return
            if name in G.SCA:
                template, scc = G.SCA[name]
                if name.endswith('_b64'):
                    if name == 's_mov_b64' and self.symbolic_move(ops):
                        return
                    self.write_pair(ops[0], template.format(*(self.vector_pair(op) for op in ops[1:])),
                                    scc=bool(scc))
                elif name in G.SCC_UNMODELLED:
                    lhs, rhs = map(self.word, ops[1:])
                    self.emit('{ uint lhs_word = %s, rhs_word = %s;' % (lhs, rhs))
                    expr = template.format('lhs_word', 'rhs_word')
                    self.write_word(ops[0], expr)
                    self.set_scc(G.add_sub_min_max_scc(name, 'lhs_word', 'rhs_word', expr), proof='0u')
                    self.emit('}')
                else:
                    self.write_word(ops[0], template.format(*(self.word(op) for op in ops[1:])), scc)
                return
            binary = re.fullmatch(r's_(and|or|xor|andn2|orn2|nand|nor|xnor)_(saveexec|wrexec)_b64', name)
            if binary and (binary[2] == 'saveexec' or binary[1] == 'andn2'):
                template, _ = G.SCA['s_%s_b64' % binary[1]]
                expr = template.format(self.vector_pair(ops[1]), 'uint2(exec_lo, exec_hi)')
                self.write_pair(ops[0], expr, scc=True, save=binary[2] == 'saveexec',
                                exec_write=binary[2] == 'wrexec')
                return
            if name in ('s_cselect_b32', 's_cselect_b64', 's_cmov_b32', 's_cmov_b64'):
                sources = ops[1:] if name.startswith('s_cselect') else [ops[1], ops[0]]
                pair = name.endswith('64')
                convert = self.vector_pair if pair else self.word
                write = self.write_pair if pair else self.write_word
                write(ops[0], 'scc ? %s : %s' % tuple(map(convert, sources)))
                return
            if name.startswith(('s_cmp_', 's_bitcmp')) and name.endswith('32'):
                operands = ['serial_word_' + op if op in MASK_WORDS | {'exec', 'vcc'} else op for op in ops]
                super().one(name, name + ' ' + ', '.join(operands))
                return
            helpers = {'s_lshl_b64': 'shl64', 's_lshr_b64': 'shr64u', 's_ashr_i64': 'shr64i',
                       's_bfe_i64': 'bfe64', 's_bfe_u64': 'bfe64'}
            if name in helpers:
                lo, hi = self.pair(ops[1])
                signed = ', 1u' if name == 's_bfe_i64' else ', 0u' if name == 's_bfe_u64' else ''
                self.write_pair(ops[0], '%s(%s, %s, %s%s)' % (helpers[name], lo, hi, self.word(ops[2]), signed),
                                scc=True)
                return
            if name == 's_bfm_b64':
                width, shift = map(self.word, ops[1:])
                self.emit('{ uint width64 = %s & 63u;' % width)
                self.write_pair(ops[0], 'shl64(lowbits(min(width64, 32u)), '
                                'lowbits(width64 - min(width64, 32u)), %s)' % shift)
                self.emit('}')
                return
            # Other pair operations can use the ordinary scalar lowering when they name real SGPRs.
            if not any(op in MASK_WORDS | {'exec', 'vcc'} for op in ops) and not any(
                    part in name for part in ('saveexec', 'wrexec', 's_wqm')):
                super().one(name, name + ' ' + ', '.join(ops))
                return
            raise G.Unsupported('wave loop: unsupported scalar ' + name)

        def symbolic_move(self, ops):
            dst, src = map(G.reg_range, ops)
            if not dst or dst[0] != 's' or dst[2] != dst[1] + 1:
                return False
            if src is None and (re.fullmatch(r'-?(?:0x[0-9a-f]+|[0-9]+)', ops[1]) or ops[1] in G.FLOATS):
                lo, hi = self.pair(ops[1])
                for n, value in zip(range(dst[1], dst[2] + 1), (lo, hi)):
                    self.set_s(n, value, ('const', int(value[:-1], 0)))
                return True
            if not src or src[0] != 's' or src[2] != src[1] + 1:
                return False
            symbols = [self.sym.get(n) for n in range(src[1], src[2] + 1)]
            if JOINED in symbols:
                raise G.Unsupported('wave loop: symbolic pair is joined across blocks')
            if any(sym and not G.run_time_symbol(sym) for sym in symbols):
                super().one('s_mov_b64', 's_mov_b64 ' + ', '.join(ops))
                return True
            self.write_pair(ops[0], self.vector_pair(ops[1]))
            for n, sym in zip(range(dst[1], dst[2] + 1), symbols):
                if sym and sym[0] == 'const':
                    self.sym[n] = sym
            return True

        def vector_pair(self, token):
            return 'uint2(%s, %s)' % self.pair(token)

        def write_word(self, dest, expr, scc=None):
            target = self.destination(dest)
            self.emit('%s = %s;' % (target, expr))
            if scc:
                self.set_scc(scc.replace('$', target), proof='0u')

        def op_x2w_serial(self, ops, mods):
            index = int(ops[0])
            _, _, name, asm = self.real[index]
            _, operands, modifiers = G.parse(asm)
            if name.startswith(NO_EFFECT):
                return
            if name == 's_branch':
                self.emit('block_pc = %du;' % self.cfg.targets[index])
            elif name in BRANCHES:
                self.emit('block_pc = (%s) ? %du : %du;' % (BRANCHES[name], self.cfg.targets[index], index + 1))
            elif name == 's_endpgm':
                self.emit('block_pc = ended;')
            elif name == 's_barrier':
                self.emit('block_pc = %du; yielded = true;' % (index + 1))
            elif name == 's_sendmsg' and operands == ['0x9']:
                self.emit('if (wave_id != 0u || alloc_seen) error = true;')
                self.emit('else { alloc_seen = true; alloc_verts = m0 & 1023u; alloc_prims = (m0 >> 12u) & 2047u; }')
            elif name.startswith(CROSS_LANE):
                self.cross_lane(name, operands, modifiers)
            elif name.startswith('v_mbcnt_'):
                self.mbcnt(name, operands, modifiers)
            elif re.fullmatch(r'ds_(?:read|write)_b(?:32|64|96|128)', name):
                self.lds(name, operands, modifiers)
            elif name == 'exp':
                _, vals, flags = G.export_operands(asm)
                if vals[1:] != ['off'] * 3 or any(f != 'done' for f in flags):
                    raise G.Unsupported('mesh primitive export form %r' % asm)
                self.emit('pword[%s] = %s;' % (LANE, self.lane_words(self.word(vals[0]))))
            elif (name.startswith(('s_getpc', 's_setpc', 's_swappc', 's_cbranch', 's_sendmsg'))
                  or '_dpp' in name or name.startswith(('ds_', 'v_permlane'))):
                raise G.Unsupported('wave loop: unsupported ' + name)
            else:
                self.scalar_native(name, operands, modifiers)

        @staticmethod
        def lane_words(line, masks=()):
            replace = {'exec': 'lexec', 'vcc': 'lvcc', 'tid': LANE}
            replace.update(('s%d' % n, 'lm%d' % n) for n in masks)
            return re.sub(r'\b(?:v\d+|exec|vcc|tid|s\d+)\b',
                          lambda m: 'v%s[%s]' % (m[0][1:], LANE) if m[0].startswith('v') and m[0] != 'vcc'
                          else replace.get(m[0], m[0]), line)

        def cross_lane(self, name, ops, mods):
            if mods or len(ops) != (2 if name == 'v_readfirstlane_b32' else 3):
                raise G.Unsupported('wave loop: invalid operands on ' + name)
            lane = ('(exec_lo != 0u ? firstbitlow(exec_lo) : '
                    '(exec_hi != 0u ? 32u + firstbitlow(exec_hi) : 0u))'
                    if name == 'v_readfirstlane_b32' else self.word(ops[2]))
            src = self.word(ops[1])
            dest = self.destination(ops[0])
            subscript = '[(wb + (cross_lane & 63u)) & 127u]'
            if re.fullmatch(r'v\d+', src):
                src += subscript
            if name == 'v_writelane_b32':
                dest += subscript
            self.emit('{ uint cross_lane = %s; uint cross_value = %s; %s = cross_value; }' % (lane, src, dest))

        def mbcnt(self, name, ops, mods):
            if mods or len(ops) != 3 or name not in ('v_mbcnt_lo_u32_b32', 'v_mbcnt_hi_u32_b32'):
                raise G.Unsupported('wave loop: invalid operands on ' + name)
            mask, base = map(self.word, ops[1:])
            dest = self.destination(ops[0])
            below = 'l - min(l, 32u)' if name.startswith('v_mbcnt_hi') else 'min(l, 32u)'
            self.emit(self.lane_words('%s = %s + countbits(%s & lowbits(%s));' % (dest, base, mask, below)))

        def lds(self, name, ops, mods):
            if len(ops) != 2 or any(not re.fullmatch(r'offset:\d+', m) for m in mods):
                raise G.Unsupported('wave loop: unsupported LDS operands')
            offset = int(mods[0].split(':')[1]) if mods else 0
            write = name.startswith('ds_write')
            value, addr = (ops[1], ops[0]) if write else ops
            words = int(name.split('_b')[1]) // 32
            dest = G.reg_range(value)
            if not dest or dest[0] != 'v' or dest[2] - dest[1] + 1 != words:
                raise G.Unsupported('wave loop: invalid LDS value ' + value)
            address = self.word(addr)
            self.emit(self.lane_words('uint base_addr = (%s + %du) >> 2u;' % (address, offset)))
            for word in range(words):
                reg = 'v%d' % (dest[1] + word)
                index = '(base_addr + %du)' % word
                array = 'lds[%s & (LDS_WORDS - 1u)]' % index
                if write:
                    self.emit(self.lane_words('if (%s < LDS_WORDS) %s = %s;' % (index, array, self.word(reg))))
                else:
                    self.emit(self.lane_words('%s = %s < LDS_WORDS ? %s : 0u;'
                                              % (self.destination(reg), index, array)))


    class SerialMeshPipeline(G.MeshPipeline):
        stage_class = SerialMeshStage

        def hlsl_ms(self):
            return self.serial_hlsl

    pipe = SerialMeshPipeline(ms, ps, root, pixel_refusal=pixel_refusal)
    # Validate the generator while this is still inside the fallback error boundary.
    try:
        pipe.serial_hlsl = render(pipe, G)
    except G.Unsupported as reason:
        reason.mesh_stage = True
        raise
    return pipe


def lane_instruction(pipe, G, name, asm, lines):
    stage = pipe.vs
    _, ops, _ = G.parse(asm)
    masks = mask_operands(G, name, ops)
    words = [('lexec', 'exec_lo', 'exec_hi'), ('lvcc', 's106', 's107')]
    words += [('lm%d' % n, 's%d' % n, 's%d' % (n + 1)) for n in masks]
    loop = ['[loop] for (uint l = 0u; l < 64u; ++l) {']
    loop.extend('uint %s = mbit(%s, %s, l);' % pair for pair in words)
    native = is_native(name, ops)
    text = []
    for line in lines:
        if name == 'exp' and not native:
            line = pipe.mesh_param_line(pipe.mesh_output_line(line))
            line = line.replace('o.', 'vbuf[%s].' % LANE).replace('mesh_vertex = true;', '')
        text.append(line if native else stage.lane_words(line, masks).replace('serial_value_', ''))
    # The shared VALU emitters predicate VGPR writes and calculate all destination mask bits.
    # Native vector work and memory accesses need an outer guard before their side effects.
    guard = native or name.startswith(('buffer_', 'tbuffer_', 'image_'))
    if guard:
        loop.append('if (lexec != 0u) {')
    loop.extend(text)
    if guard:
        loop.append('}')
    for bit, lo, hi in words:
        if any(re.search(r'\b%s\s*=(?!=)' % bit, line) for line in text):
            loop.append('mset(%s, %s, l, %s);' % (lo, hi, bit))
    return loop + ['}']


def render(pipe, G):
    s = pipe.vs
    blocks = {}
    for _, statements, _ in s.blocks:
        for line in statements:
            if line.startswith('//@B '):
                block = blocks.setdefault(int(line[5:]), {})
            elif line.startswith('//@I '):
                current = block.setdefault(int(line[5:]), [])
            else:
                current.append(line)
    hlsl = ['// x2w_serial_wave 1'] + pipe.common() + [
        'uint mbit(uint lo, uint hi, uint l) { return ((l < 32u ? lo : hi) >> (l & 31u)) & 1u; }',
        'void mset(inout uint lo, inout uint hi, uint l, uint value) {',
        '    uint bit = 1u << (l & 31u);',
        '    if (l < 32u) lo = (lo & ~bit) | ((value & 1u) << (l & 31u));',
        '    else hi = (hi & ~bit) | ((value & 1u) << (l & 31u));',
        '}',
        'uint lowbits(uint n) { return n >= 32u ? 0xffffffffu : (1u << (n & 31u)) - 1u; }',
        'struct VSOut {',
    ] + pipe.interface(False) + [
        '};',
        'static const uint LDS_WORDS = 1024u;',
        '[numthreads(1, 1, 1)]',
        '[outputtopology("triangle")]',
        'void main(uint3 gid : SV_GroupID, out vertices VSOut verts[128], out indices uint3 tris[128]) {',
        '    uint lds[LDS_WORDS];',
        '    VSOut vbuf[128]; uint pword[128];',
    ]
    vgprs = ['v%d' % n for n in sorted(s.used_v | {5})]
    scalars = ['s%d' % n for n in sorted(s.used_s)]
    # Mask-only uses reference both words, including the high word absent from a single-lane lowering.
    hi_words = set()
    for block in blocks.values():
        for index, lines in block.items():
            _, _, name, asm = s.real[index]
            if not lines or lines[0].startswith('// X2W_SERIAL_STUB'):
                continue
            for n in mask_operands(G, name, G.parse(asm)[1]):
                hi_words.update((n, n + 1))
    scalars = sorted(set(scalars) | {'s%d' % n for n in hi_words | {106, 107}})
    words = scalars + ['exec_lo', 'exec_hi', 'm0']
    saved = words + ['scc']
    hlsl.extend('    uint %s[128];' % name for name in vgprs)
    hlsl.extend('    %s saved_%s[2];' % ('bool' if name == 'scc' else 'uint', name) for name in saved)
    hlsl += [
        '    uint pc[2] = {0u, 0u};',
        '    const uint ended = 0xffffffffu;',
        '    uint alloc_verts = 0u, alloc_prims = 0u, fuel = 65536u;',
        '    bool alloc_seen = false, error = false;',
        '    [loop] for (uint i = 0u; i < LDS_WORDS; ++i) lds[i & (LDS_WORDS - 1u)] = 0u;',
        '    [loop] for (uint i = 0u; i < 128u; ++i) {',
        '        vbuf[i & 127u] = (VSOut)0; pword[i & 127u] = 0u;',
    ]
    # The proof path models entry v5 >> 7 at the writer as the group ID.
    # Give that same entry value to every lane of both guest waves.
    hlsl.extend('        %s[i & 127u] = %s;' % (name, 'gid.x << 7u' if name == 'v5' else '0u') for name in vgprs)
    hlsl += [
        '    }',
        '    [loop] for (uint wave_id = 0u; wave_id < 2u; ++wave_id) {',
    ]
    hlsl.extend('        uint %s = %s;' % (name, '0xffffffffu' if name.startswith('exec_') else '0u')
                for name in words)
    hlsl.append('        bool scc = false;')
    hlsl.extend('        ' + line for line in s.entry)
    for n, symbol in sorted(s.entry_symbols.items()):
        if symbol[0] == 'ud' and 's%d' % n in scalars:
            hlsl.append('        s%d = ud%d;' % (n, symbol[1]))
    hlsl.extend('        saved_%s[wave_id & 1u] = %s;' % (name, name) for name in saved)
    # A round resumes each live wave once. Yielding to the next round pairs barriers by count.
    hlsl += [
        '    }',
        '    [loop] while (!error && (pc[0] != ended || pc[1] != ended)) {',
        '        bool progress = false;',
        '        [loop] for (uint wave_id = 0u; wave_id < 2u && !error; ++wave_id) {',
        '            uint block_pc = pc[wave_id & 1u];',
        '            if (block_pc == ended) continue;',
        '            uint wb = wave_id * 64u;',
        '            bool yielded = false;',
    ]
    hlsl.extend('            %s %s = saved_%s[wave_id & 1u];' %
                ('bool' if name == 'scc' else 'uint', name, name) for name in saved)
    hlsl += [
        '            uint a = 0u, w = 0u; float4 t = 0;',
    ]
    if getattr(s, 'used_scratch', False):
        hlsl.append('            uint discard_word = 0u;')
    hlsl.extend('            uint %s = 0u;' % name for name in sorted(s.uniform_load_temps))
    if 'uint' in s.used_atomic:
        hlsl.append('            uint ao = 0u;')
    if 'int' in s.used_atomic:
        hlsl.append('            int ai = 0;')
    hlsl += [
        '            [loop] while (!yielded && !error && block_pc != ended) {',
        '                if (fuel == 0u) { error = true; break; }',
        '                --fuel; progress = true;',
        '                switch (block_pc) {',
    ]
    for start, operations in blocks.items():
        hlsl.append('                case %du: {' % start)
        for index, lines in operations.items():
            _, _, name, asm = s.real[index]
            if index in s.errors:
                lines = ['// X2W_SERIAL_STUB %s %s' % (name, ' '.join(str(s.errors[index]).split()))]
                if is_branch(name) or name in ('s_endpgm', 's_barrier'):
                    lines.append('error = true;')
            if lines and not lines[0].startswith('// X2W_SERIAL_STUB') and name.startswith(
                    ('v_', 'buffer_', 'tbuffer_', 'image_', 'exp', 'ds_')) and not name.startswith(CROSS_LANE):
                lines = lane_instruction(pipe, G, name, asm, lines)
            hlsl.extend('                    ' + line for line in lines)
        end = s.cfg.blocks[start]
        last = s.real[end - 1][2]
        if not is_branch(last) and last not in ('s_endpgm', 's_barrier'):
            hlsl.append('                    block_pc = %s;' % ('%du' % end if end in blocks else 'ended'))
        hlsl += ['                    break;', '                }']
    hlsl += [
        '                default: error = true; break;',
        '                }',
        '            }',
        '            pc[wave_id & 1u] = block_pc;',
    ]
    hlsl.extend('            saved_%s[wave_id & 1u] = %s;' % (name, name) for name in saved)
    hlsl += [
        '        }',
        '        if (!progress) error = true;',
        '    }',
        '    uint nv = (alloc_seen && !error) ? min(alloc_verts, 128u) : 0u;',
        '    uint np = nv == 0u ? 0u : min(alloc_prims, 128u);',
        '    SetMeshOutputCounts(nv, np);',
        '    [loop] for (uint i = 0u; i < nv; ++i) verts[i & 127u] = vbuf[i & 127u];',
        '    [loop] for (uint i = 0u; i < np; ++i) {',
        '        uint word = pword[i & 127u];',
        '        tris[i & 127u] = word >> 31u != 0u ? (uint3)0u :',
        '            min(uint3(word, word >> 10u, word >> 20u) & 511u, nv - 1u);',
        '    }',
        '}',
    ]
    text = '\n'.join(hlsl) + '\n'
    code = re.sub(r'//[^\n]*', '', text)
    if leaked := LEAK.search(code):
        raise G.Unsupported('wave loop: generator leaked ' + leaked[0])
    if re.search(r'\[unroll\]', code):
        raise G.Unsupported('wave loop: generator leaked unroll')
    return text
