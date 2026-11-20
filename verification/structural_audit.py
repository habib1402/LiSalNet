#!/usr/bin/env python3
"""Source-driven symbolic audit, NOT PyTorch execution or a latency benchmark.

Reads the supplied class definitions unchanged, substitutes shape/count operators,
and traces their Python forward control flow. No weights, datasets or GPU needed.
Counts unique registered parameters and Conv2d MACs, including repeated calls.
Usage: python structural_audit.py /path/to/upload --output structural_results.json
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

TRACE = []
NAMES = {}

def pair(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x, x)

class Tensor:
    def __init__(self, shape, deps=frozenset()):
        self.shape = tuple(shape)
        self.deps = frozenset(deps)
    def combine(self, other):
        if not isinstance(other, Tensor):
            return self
        assert len(self.shape) == len(other.shape)
        assert all(a == b or a == 1 or b == 1 for a, b in zip(self.shape, other.shape)), (self.shape, other.shape)
        return Tensor(tuple(max(a, b) for a, b in zip(self.shape, other.shape)), self.deps | other.deps)
    __add__ = combine
    __mul__ = combine

class Parameter:
    def __init__(self, n):
        self.n = n
    def numel(self):
        return self.n

class Module:
    def __init__(self):
        self._own_params = []
        self._buffer_bytes = 0
    def named_children(self):
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value
    def named_modules(self, prefix='', seen=None):
        if seen is None:
            seen = set()
        if id(self) in seen:
            return
        seen.add(id(self))
        yield prefix, self
        for name, module in self.named_children():
            yield from module.named_modules('.'.join(filter(None, [prefix, name])), seen)
    def modules(self):
        return (m for _, m in self.named_modules())
    def parameters(self):
        return (p for m in self.modules() for p in m._own_params)
    def __call__(self, *args):
        out = self.forward(*args)
        TRACE.append({'name': NAMES.get(id(self), self.__class__.__name__), 'type': self.__class__.__name__,
                      'input': shapes(args), 'output': shapes(out)})
        return out

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self._layers = []
        for layer in layers:
            self.append(layer)
    def append(self, layer):
        setattr(self, str(len(self._layers)), layer)
        self._layers.append(layer)
    def __iter__(self):
        return iter(self._layers)
    def __getitem__(self, i):
        return self._layers[i]
    def __len__(self):
        return len(self._layers)
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.ci, self.co, self.k = in_channels, out_channels, pair(kernel_size)
        self.s, self.p, self.d, self.g = pair(stride), pair(padding), pair(dilation), groups
        assert in_channels % groups == out_channels % groups == 0
        self.weight = Parameter(out_channels * (in_channels // groups) * math.prod(self.k))
        self.bias = Parameter(out_channels) if bias else None
        self._own_params = [self.weight] + ([self.bias] if bias else [])
        self.macs = 0
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.ci, (NAMES.get(id(self)), x.shape, self.ci)
        oh, ow = [(v + 2*p - d*(k-1) - 1)//s + 1 for v,p,d,k,s in zip((h,w), self.p, self.d, self.k, self.s)]
        self.macs += b * oh * ow * self.weight.n
        return Tensor((b, self.co, oh, ow), x.deps | {id(p) for p in self._own_params})

class BatchNorm2d(Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.weight, self.bias = Parameter(c), Parameter(c)
        self._own_params = [self.weight, self.bias]
        self._buffer_bytes = 8*c + 8  # float32 running mean/var, int64 batch counter
        self.output_elements = 0
    def forward(self, x):
        assert x.shape[1] == self.c
        self.output_elements += math.prod(x.shape)
        return Tensor(x.shape, x.deps | {id(p) for p in self._own_params})

class GroupNorm(BatchNorm2d):
    def __init__(self, groups, c):
        super().__init__(c)
        assert c % groups == 0
        self._buffer_bytes = 0

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Pool(Identity):
    def __init__(self, output_size):
        super().__init__()
        self.size = pair(output_size)
    def forward(self, x):
        return Tensor((*x.shape[:2], *self.size), x.deps)

def cat(xs, dim):
    shape = list(xs[0].shape)
    assert all(all(a == b for i, (a,b) in enumerate(zip(shape, x.shape)) if i != dim) for x in xs)
    shape[dim] = sum(x.shape[dim] for x in xs)
    return Tensor(shape, frozenset().union(*(x.deps for x in xs)))

def interpolate(x, size, **kwargs):
    return Tensor((*x.shape[:2], *pair(size)), x.deps)

def shapes(x):
    if isinstance(x, Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [shapes(v) for v in x]
    return None

def load_definitions(path, extra=None):
    # Only class/function definitions and literal model_urls assignment are needed.
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    noop = lambda *args, **kwargs: None
    nn = SimpleNamespace(Module=Module, Sequential=Sequential, ModuleList=Sequential,
                         Conv2d=Conv2d, BatchNorm2d=BatchNorm2d, GroupNorm=GroupNorm,
                         ReLU=Identity, ReLU6=Identity, LeakyReLU=Identity, PReLU=Identity,
                         Sigmoid=Identity, Dropout2d=Identity, Identity=Identity,
                         AdaptiveAvgPool2d=Pool, AdaptiveMaxPool2d=Pool, Linear=type('Linear', (), {}),
                         init=SimpleNamespace(kaiming_normal_=noop, zeros_=noop, ones_=noop, normal_=noop))
    env = {'nn': nn, 'torch': SimpleNamespace(cat=cat, sigmoid=lambda x: x), 'math': math,
           'F': SimpleNamespace(relu=lambda x: x, gelu=lambda x: x, interpolate=interpolate)}
    env.update(extra or {})
    exec(compile(tree, str(path), 'exec'), env)
    return env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backbone = load_definitions(args.source_dir / 'MobileNetV2.py')
    net = load_definitions(args.source_dir / 'model.py', {'mobilenet_v2': backbone['mobilenet_v2']})
    model = net['LiSalNet'](pretrained=False)
    NAMES.update({id(m): name for name, m in model.named_modules()})
    parameters = list(model.parameters())
    result = {'verification': 'Symbolic shapes, parameter formulas and Python forward control flow; no numerical PyTorch execution.',
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.source_dir.glob('*')) if p.suffix in ['.py','.sh']},
              'parameters': sum(p.n for p in parameters),
              'modules': {name: sum(p.n for p in m.parameters()) for name,m in model.named_children()},
              'buffer_bytes': sum(m._buffer_bytes for m in model.modules()),
              'runs': {}}
    result['parameter_MiB_FP32'] = result['parameters']*4 / 2**20
    result['parameter_and_buffer_MiB'] = (result['parameters']*4 + result['buffer_bytes'])/2**20
    for size in [256,288,320]:
        TRACE.clear()
        for m in model.modules():
            if isinstance(m, Conv2d):
                m.macs = 0
        y = model(Tensor((1,3,size,size)), Tensor((1,1,size,size)))
        unused = [{'name': name, 'parameters':sum(p.n for p in m._own_params)}
                  for name,m in model.named_modules() if m._own_params and all(id(p) not in y.deps for p in m._own_params)]
        result['runs'][str(size)] = {'output_shape': shapes(y),
                    'conv_MACs': sum(m.macs for m in model.modules() if isinstance(m,Conv2d)),
                    'conv_MACs_by_module': {name: sum(m.macs for m in child.modules() if isinstance(m,Conv2d)) for name,child in model.named_children()},
                    'parameters_outside_prediction_dependencies': unused,
                    'trace': list(TRACE)}
        bn_elements = sum(math.prod(t['output']) for t in TRACE if t['type'] == 'BatchNorm2d')
        result['runs'][str(size)]['BN_output_elements'] = bn_elements
        result['runs'][str(size)]['illustrative_conv_MACs_plus_4_per_BN_element'] = (
            result['runs'][str(size)]['conv_MACs'] + 4*bn_elements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','source_sha256']},indent=2))
    for size,run in result['runs'].items():
        print(size, 'conv_MACs=',run['conv_MACs'], 'unused=',run['parameters_outside_prediction_dependencies'])

if __name__ == '__main__':
    main()
