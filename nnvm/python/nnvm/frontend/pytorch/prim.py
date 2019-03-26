r'''This file contains one class per PyTorch Prim operator. For the full list
of operators, see
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/interned_strings.h
'''
import re
import tvm
import numpy as np
from .base import PyTorchOp


class PrimOp(PyTorchOp):
    r'''Base class for Prim operators'''


class Constant(PrimOp):
    r'''prim::Constant operator'''

    def __init__(self, node, graph):
        super(Constant, self).__init__(node, graph)
        output = next(node.outputs())
        type_kind = output.type().kind()
        value = self._parse_value_from_string()
        output_name = self.get_output_name(0)
        if type_kind == 'IntType':
            self.set_output(0, output_name, int(value))
        elif type_kind == 'FloatType':
            self.set_output(0, output_name, value)
        elif type_kind == 'BoolType':
            self.set_output(0, output_name, bool(value))
        elif type_kind == 'CompleteTensorType' and output.type().sizes() == []:
            self.shape = output.type().sizes()
            arr = value * np.ones(self.shape).astype(float)
            self.set_output(0, output_name, tvm.nd.array(arr))
        elif type_kind == 'StringType':
            self.set_output(0, output_name, value)
        else:
            msg = 'Only "IntType", "FloatType", "BoolType", "StringType", and ' \
                  '"CompleteTensorType" type-kinds are supported. For ' \
                  '"CompleteTensorType", type-sizes must be [].'
            raise RuntimeError(msg)

    def _parse_value_from_string(self):
        r'''For some reason, __getitem__ is sometimes stripped from the
        torch._C.Node objects.'''
        pattern = r'(?<=value=)[^]]+'
        string = str(self._node)
        value_string = re.findall(pattern, string)[0].strip('{}')
        try:
            return float(value_string)
        except ValueError:
            return None


class ListConstruct(PrimOp):
    r'''prim::ListConstruct operator'''

    def __init__(self, node, graph):
        super(ListConstruct, self).__init__(node, graph)
        self.set_output(0, self.get_output_name(0),
                        [inp.get_output(0) for inp in self.inputs])


class Int(PrimOp):
    r'''prim::Int operator'''

    def __init__(self, node, graph):
        super(Int, self).__init__(node, graph)
        val = self.get_input(0).get_output(0).asnumpy()
        self.set_output(0, self.get_output_name(0), int(val))


class NumToTensor(PrimOp):
    r'''prim::NumToTensor operator'''

    def __init__(self, node, graph):
        super(NumToTensor, self).__init__(node, graph)
        self.shape = []
        val = self.get_input(0).get_output(0)
        dtype = type(val)
        arr = val * np.ones(self.shape).astype(dtype)
        self.set_output(0, self.get_output_name(0), tvm.nd.array(arr))


class Undefined(PrimOp):
    r'''prim::Undefined operator'''

    def __init__(self, node, graph):
        super(Undefined, self).__init__(node, graph)
        self.set_output(0, self.get_output_name(0), None)

class ListUnpack(PrimOp):
    r'''prim::ListUnpack operator'''

    def __init__(self, node, graph):
        super(ListUnpack, self).__init__(node, graph)
        for i in range(len(self.output_names)):
            self.set_output(i, self.get_output_name(i),
                            self.get_input(0).get_output(0)[i])


PRIM_MAP = {
    'Constant': Constant,
    'ListConstruct': ListConstruct,
    'TupleConstruct': ListConstruct,
    'Int': Int,
    'NumToTensor': NumToTensor,
    'Undefined': Undefined,
    'ListUnpack': ListUnpack,
}
