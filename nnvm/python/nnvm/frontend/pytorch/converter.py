r'''Convert PyTorch models to NNVM symbol graphs'''
from pickle import UnpicklingError
import tvm
from nnvm.symbol import Symbol, Group
import numpy as np
import torch
from .aten import ATEN_MAP
from .prim import PRIM_MAP
from .base import PyTorchGraph


def operator_map(kind):
    namespace, op_name = kind.split('::')
    return {
        'aten': ATEN_MAP,
        'prim': PRIM_MAP,
    }[namespace][op_name]


class PyTorchConverter:
    r'''Converter from PyTorch JIT IR to NNVM'''

    def __init__(self, filename, input_shapes):
        self._load_model(filename, input_shapes)
        self._num_inputs = len(input_shapes)
        self.graph = PyTorchGraph()
        self._parse_inputs(input_shapes)
        self._parse_params()
        self._parse_ops()

    def _load_model(self, filename, input_shapes):
        try:
            self._trace = torch.jit.load(filename).float().eval()
        except RuntimeError:
            try:
                self._trace = torch.load(filename).float().eval()
            except UnpicklingError:
                raise RuntimeError('Failed to load model')
        shapes = [input_shapes[k] for k in sorted(input_shapes)]
        inputs = [torch.zeros(shape).float() for shape in shapes]
        try:
            self._trace = torch.jit.trace(self._trace, *inputs).float().eval()
        except RuntimeError:
            inputs = [inp.cuda() for inp in inputs]
            self._trace = torch.jit.trace(self._trace, *inputs).float().eval().cpu()
            inputs = [inp.cpu() for inp in inputs]
            self._trace = torch.jit.trace(self._trace, *inputs).float().eval().cpu()
        print(self._trace.graph)

    @property
    def _ir_tensor_names(self):
        return [i.uniqueName() for i in self._trace.graph.inputs()]

    def _parse_inputs(self, input_shapes):
        input_names = sorted(input_shapes)
        ir_names = self._ir_tensor_names[:self._num_inputs]
        ir_name_map = dict(zip(input_names, ir_names))
        inv_ir_name_map = dict((v, k) for k, v in ir_name_map.items())
        for i, inp in enumerate(self._trace.graph.inputs()):
            if i >= self._num_inputs:
                break
            ir_name = inp.uniqueName()
            if ir_name in inv_ir_name_map:
                inp.setUniqueName(inv_ir_name_map[ir_name])
        for input_name in sorted(input_shapes):
            input_shape = input_shapes[input_name]
            tensor = np.zeros(input_shape).astype(np.float32)
            ir_name = ir_name_map[input_name]
            for inp in self._trace.graph.inputs():
                if inp.uniqueName() == ir_name:
                    inp.setUniqueName(input_name)
                    break
            self.graph.add_input(input_name, tensor)

    def _parse_params(self):
        state_dict = self._trace.state_dict()
        ignored_params = [
#            'num_batches_tracked',
        ]
        state_dict_names = []
        for k in state_dict.keys():
            if not any(ignored_param in k for ignored_param in ignored_params):
               state_dict_names.append(k)
        ir_names = self._ir_tensor_names[self._num_inputs:]
#        ir_names.reverse()
        name_map = dict(zip(state_dict_names, ir_names))
#        state_dict_names = state_dict_names[:len(ir_names)]
        for state_dict_name in state_dict_names:
            param = state_dict[state_dict_name]
            ir_name = name_map[state_dict_name]
            tensor = param.cpu().numpy()
            self.graph.add_param(ir_name, tensor)

    def _parse_ops(self):
        unsupported_ops = set()
        for node in self._trace.graph.nodes():
            kind = node.kind()
            try:
                operator_map(kind)
            except KeyError:
                unsupported_ops.add(kind)
        if unsupported_ops:
            ops_str = str(list(unsupported_ops)).strip('[]').replace("'", '')
            msg = 'The following operators are not implemented: {}'
            raise tvm.error.OpNotImplemented(msg.format(ops_str))
        from collections import defaultdict
        kinds = defaultdict(int)
        for node in self._trace.graph.nodes():
            kind = node.kind()
            self.graph.add_op(operator_map(kind)(node, self.graph))
            kinds[kind] += 1
        for kind, count in kinds.items():
            print(f'{kind}: {count}')

    def convert(self):
        r'''Convert the parsed PyTorch model to an NNVM symbol graph and
        parameter dict.'''
        params = {k: tvm.nd.array(v.arr) for k, v in self.graph.params.items()}
        incoming_nodes = set()
        for name, op in self.graph.ops.items():
            incoming_nodes.update(op.input_names)
        outputs = []
        for name in self.graph.ops:
            for i in range(len(self.graph.ops[name].outputs)):
                output_name = self.graph.ops[name].get_output_name(i)
                node = self.graph.ops[name].get_output(i)
                if output_name not in incoming_nodes:
                    output = node.get_output(0)
                    if isinstance(output, Symbol):
                        outputs.append(output)
                    elif isinstance(output, list):
                        is_symbol = lambda n: isinstance(n, Symbol)
                        outputs.extend(filter(is_symbol, output))
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = Group(outputs)
        return output, params


def from_pytorch(filename, input_shapes):
    converter = PyTorchConverter(filename, input_shapes)
    sym, params = converter.convert()
    return sym, params
