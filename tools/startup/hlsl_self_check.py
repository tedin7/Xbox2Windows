"""Check names and returned output structs in the translator's HLSL subset."""

import re


TOKEN = re.compile(
    r'//[^\n]*|\#[^\n]*|"[^"\n]*"|0x[\da-fA-F]+[uUlL]*|'
    r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[fFuUlL]*|'
    r'[a-zA-Z_]\w*|==|!=|<=|>=|[{}()\[\],;:.?=<>]|[^\s]')
TYPE = re.compile(r'(?:bool|int|uint|float|half|double|min16float|min16uint|min16int)[1-4]?(?:x[1-4])?'
                  r'|(?:u?int64_t|float16_t)(?:[1-4])?|void|(?:RW)?(?:ByteAddressBuffer|StructuredBuffer|Buffer|'
                  r'Texture(?:1D|2D|3D|Cube)(?:Array|MS|MSArray)?)|Sampler(?:Comparison)?State')
QUALIFIERS = frozenset('const static precise groupshared uniform out in inout vertices indices '
                       'nointerpolation linear centroid sample noperspective'.split())
INTRINSICS = frozenset('abs all any asfloat asint asuint ceil clamp countbits ddx ddx_fine ddy ddy_fine dot exp2 f16tof32 f32tof16 '
                       'firstbithigh firstbitlow floor frac isfinite isinf isnan lerp log2 mad max min pow rcp '
                       'reversebits round rsqrt saturate sign sin cos sqrt trunc '
                       'WaveActiveAllTrue WaveActiveAnyTrue WaveActiveBallot WaveActiveBitAnd WaveActiveBitOr '
                       'WaveActiveBitXor WaveActiveCountBits WaveActiveMax WaveActiveMin WaveActiveSum '
                       'WaveGetLaneCount WaveGetLaneIndex WaveIsFirstLane WaveReadLaneAt WaveReadLaneFirst '
                       'QuadReadAcrossX QuadReadAcrossY QuadReadAcrossDiagonal QuadReadLaneAt '
                       'GroupMemoryBarrierWithGroupSync DeviceMemoryBarrierWithGroupSync GroupMemoryBarrier '
                       'DeviceMemoryBarrier AllMemoryBarrierWithGroupSync AllMemoryBarrier '
                       'InterlockedAdd InterlockedAnd InterlockedCompareExchange InterlockedExchange '
                       'InterlockedMax InterlockedMin InterlockedOr InterlockedXor SetMeshOutputCounts '
                       'NonUniformResourceIndex'.split())
IDENTIFIER = re.compile(r'[a-zA-Z_]\w*')
ORIGIN = re.compile(r' // x2w-origin (?P<offset>0x[0-9a-f]+) (?P<instruction>[a-z][a-z0-9_]*) '
                    r'(?P<handler>[a-z_][a-z0-9_]*)$', re.MULTILINE)


def check_hlsl(source):
    """Return the first diagnostic, or None. Track lexical scopes and definite struct writes."""
    try:
        Check(source).program()
    except InvalidEmission as error:
        return str(error)
    return None


class InvalidEmission(Exception):
    pass


class Check:
    def __init__(self, source):
        self.tokens = []
        self.origins = {}
        self.scopes = [{}]
        self.shadowed_outputs = []
        self.structs = {}
        self.index = 0
        self.outputs = {}
        self.return_type = 'void'
        self.mesh_outputs = {}
        for row, line in enumerate(source.splitlines()):
            origin = ORIGIN.search(line)
            if origin:
                self.origins[row] = origin['instruction']
            macro = re.match(r'#define\s+(\w+)', line)
            if macro:
                self.scopes[0][macro[1]] = None
            self.tokens.extend((t, row) for t in TOKEN.findall(line) if not t.startswith(('//', '#')))
        self.tokens.append(('', len(source.splitlines())))

    def peek(self):
        return self.tokens[self.index][0]

    def take(self):
        token = self.peek()
        if not token:
            self.fail('unexpected end of emitted HLSL')
        self.index += 1
        return token

    def accept(self, token):
        if self.peek() == token:
            self.index += 1
            return True
        return False

    def expect(self, token):
        if not self.accept(token):
            self.fail(f'expected {token} in emitted HLSL')

    def fail(self, text, row=None):
        if row is None:
            row = self.tokens[self.index][1]
        raise InvalidEmission(f'internal: {text} (emitted for {self.origins.get(row, "stage_wrapper")})')

    def is_type(self, token):
        return token in self.structs or TYPE.fullmatch(token)

    def typename(self):
        while self.peek() in QUALIFIERS:
            self.take()
        kind = self.take()
        if not self.is_type(kind):
            self.fail(f'{kind} used but not declared', self.tokens[self.index - 1][1])
        if self.accept('<'):
            self.typename()
            self.expect('>')
        return kind

    def visible_name(self, token):
        return any(token in scope for scope in reversed(self.scopes))

    def expression(self, end):
        parts = []
        while self.peek() not in end:
            row = self.tokens[self.index][1]
            token = self.take()
            if token in ('(', '['):
                close = ')' if token == '(' else ']'
                parts.extend([token, *self.expression({close})])
                self.expect(close)
                parts.append(close)
                continue
            if parts[-1:] == ['.'] and len(parts) >= 2:
                kind = next((scope[parts[-2]] for scope in reversed(self.scopes) if parts[-2] in scope), None)
                if kind in self.structs and token not in self.structs[kind]:
                    self.fail(f'{token} used but not declared', row)
            if (IDENTIFIER.fullmatch(token) and parts[-1:] != ['.'] and not self.is_type(token)
                    and token not in INTRINSICS and token not in ('true', 'false')
                    and not self.visible_name(token)):
                self.fail(f'{token} used but not declared', row)
            parts.append(token)
        return parts

    def annotation(self):
        if self.accept(':'):
            self.take()
            if self.accept('('):
                self.take()
                while self.accept(','):
                    self.take()
                self.expect(')')

    def defined_fields(self, value, kind):
        if len(value) == 1:
            return self.outputs.get(value[0], set()).copy()
        if value == ['(', kind, ')', '0']:
            return set(self.structs[kind])
        if value[1:2] == ['('] and value[-1:] == [')'] and self.scopes[0].get(value[0]) == kind:
            return set(self.structs[kind])
        return set()

    def variable(self, kind, name, end):
        if self.shadowed_outputs and name in self.outputs and name not in self.scopes[-1]:
            self.shadowed_outputs[-1][name] = self.outputs.pop(name)
        self.scopes[-1][name] = kind
        array = self.accept('[')
        if array:
            self.expression({']'})
            self.expect(']')
        self.annotation()
        value = self.expression(end) if self.accept('=') else []
        if kind in self.structs and len(self.scopes) > 1 and not array:
            self.outputs[name] = self.defined_fields(value, kind)

    def declarations(self, kind, name):
        self.variable(kind, name, {',', ';'})
        while self.accept(','):
            self.variable(kind, self.take(), {',', ';'})
        self.expect(';')

    def attributes(self):
        while self.accept('['):
            while not self.accept(']'):
                self.take()

    def enter_scope(self):
        self.scopes.append({})
        self.shadowed_outputs.append({})

    def leave_scope(self):
        for name in self.scopes.pop():
            self.outputs.pop(name, None)
        self.outputs.update(self.shadowed_outputs.pop())

    def block(self):
        self.expect('{')
        self.enter_scope()
        live = True
        while not self.accept('}'):
            if live:
                live = self.statement()
            else:
                # Unreachable text still has to compile.
                self.statement()
        self.leave_scope()
        return live

    def scoped_statement(self):
        self.enter_scope()
        live = self.statement()
        self.leave_scope()
        return live

    def statement(self):
        self.attributes()
        if self.peek() == '{':
            return self.block()
        if self.accept('if'):
            self.expect('(')
            self.expression({')'})
            self.expect(')')
            before = {name: fields.copy() for name, fields in self.outputs.items()}
            then_live = self.scoped_statement()
            then, self.outputs = self.outputs, before
            else_live = self.scoped_statement() if self.accept('else') else True
            if then_live:
                self.outputs = ({name: fields & then[name] for name, fields in self.outputs.items()}
                                if else_live else then)
            return then_live or else_live
        if self.accept('switch'):
            self.expect('(')
            self.expression({')'})
            self.expect(')')
            self.expect('{')
            before = {name: fields.copy() for name, fields in self.outputs.items()}
            self.enter_scope()
            while not self.accept('}'):
                if self.accept('case'):
                    self.outputs = {name: fields.copy() for name, fields in before.items()}
                    self.expression({':'})
                    self.expect(':')
                elif self.accept('default'):
                    self.outputs = {name: fields.copy() for name, fields in before.items()}
                    self.expect(':')
                else:
                    self.statement()
            # The emitted format selectors write scalar temporaries. No switch write
            # establishes a definite assignment of an output outside the selector.
            self.leave_scope()
            self.outputs = before
            return True
        if self.accept('for'):
            self.enter_scope()
            self.expect('(')
            self.statement()
            self.expression({';'})
            self.expect(';')
            self.expression({')'})
            self.expect(')')
            before = {name: fields.copy() for name, fields in self.outputs.items()}
            self.scoped_statement()
            self.outputs = before
            self.leave_scope()
            return True
        row = self.tokens[self.index][1]
        if self.accept('return'):
            value = self.expression({';'})
            self.expect(';')
            if self.return_type in self.structs:
                name = ''.join(value)
                missing = self.structs[self.return_type] - self.defined_fields(value, self.return_type)
                if missing:
                    self.fail(f'{name}.{min(missing)} is not assigned on every output path', row)
            return False
        if self.peek() in ('discard', 'break', 'continue'):
            self.take()
            self.expect(';')
            return False
        if not self.peek():
            self.fail('unexpected end of emitted HLSL', row)
        next_token = self.tokens[self.index + 1][0]
        if self.peek() in QUALIFIERS or (self.is_type(self.peek()) and IDENTIFIER.fullmatch(next_token)):
            self.declarations(self.typename(), self.take())
        else:
            value = self.expression({';'})
            self.expect(';')
            if value and value[0] in self.mesh_outputs and '=' in value:
                kind = self.mesh_outputs[value[0]]
                rhs = value[value.index('=') + 1:]
                missing = self.structs[kind] - self.defined_fields(rhs, kind)
                if missing:
                    self.fail(f'{value[0]}.{min(missing)} is not assigned on every output path', row)
            if value and value[0] in self.outputs:
                if value[1:2] == ['=']:
                    kind = next(scope[value[0]] for scope in reversed(self.scopes) if value[0] in scope)
                    self.outputs[value[0]] = self.defined_fields(value[2:], kind)
                elif len(value) >= 4 and value[1] == '.' and value[3] == '=':
                    self.outputs[value[0]].add(value[2])
        return True

    def program(self):
        while self.peek():
            self.attributes()
            if self.accept('struct'):
                name = self.take()
                fields = self.structs[name] = set()
                self.expect('{')
                while not self.accept('}'):
                    self.typename()
                    fields.add(self.take())
                    self.annotation()
                    self.expect(';')
                self.expect(';')
            elif self.accept('cbuffer'):
                self.take()
                self.annotation()
                self.expect('{')
                while not self.accept('}'):
                    self.declarations(self.typename(), self.take())
                self.expect(';')
            else:
                kind, name = self.typename(), self.take()
                if not self.accept('('):
                    self.declarations(kind, name)
                    continue
                self.scopes[0][name] = kind
                self.enter_scope()
                self.outputs = {}
                self.mesh_outputs = {}
                self.return_type = kind
                if not self.accept(')'):
                    while True:
                        mesh_vertex = self.peek() == 'out' and self.tokens[self.index + 1][0] == 'vertices'
                        param_kind, param = self.typename(), self.take()
                        self.variable(param_kind, param, {',', ')'})
                        if mesh_vertex and param_kind in self.structs:
                            self.mesh_outputs[param] = param_kind
                        if not self.accept(','):
                            break
                    self.expect(')')
                self.annotation()
                if self.block() and kind in self.structs:
                    self.fail('output struct is not returned on every path')
                self.leave_scope()
