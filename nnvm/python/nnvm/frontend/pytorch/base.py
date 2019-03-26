r'''Basic classes for PyTorch operators and graphs'''
from collections import OrderedDict
from nnvm.symbol import Variable, Symbol
from nnvm.frontend.common import get_nnvm_op
from nnvm.compiler.graph_util import infer_shape
from nnvm.graph import create


def make_symbol(topi_name, *inputs, **attrs):
    r'''Create an NNVM symbol given a Topi name, inputs, and attrs.'''
    return get_nnvm_op(topi_name)(*inputs, **attrs)


def attr_2d(val, default=None):
    r'''Helper function for computing attributes of 2D functions'''
    if not val:
        return [default] * 2
    if isinstance(val, list):
        return val
    return [int(val)] * 2


class PyTorchGraph:
    r'''Wrapper for the PyTorch JIT IR graph'''

    def __init__(self):
        self.inputs = OrderedDict()
        self.params = OrderedDict()
        self.ops = OrderedDict()
        self.outputs = OrderedDict()

    def __getitem__(self, name):
        if name in self.inputs:
            return self.inputs[name]
        if name in self.params:
            return self.params[name]
        if name in self.ops:
            return self.ops[name]
        if name in self.outputs:
            return self.outputs[name]
        raise RuntimeError('Node {} not found.'.format(name))

    def __contains__(self, name):
        attrs = ['inputs', 'params', 'ops', 'outputs']
        return any(name in getattr(self, k) for k in attrs)

    def add_input(self, name, tensor):
        r'''Add an input of the PyTorch model'''
        self.inputs[name] = PyTorchInput(name, tensor, self)

    def add_param(self, name, tensor):
        r'''Add a param of the PyTorch model'''
        self.params[name] = PyTorchParam(name, tensor.astype('float32'), self)

    def add_op(self, op_node):
        r'''Add an operator and its associated outputs of the PyTorch model'''
        self.ops[op_node.name] = op_node
        for i in range(len(op_node.outputs)):
            self.outputs[op_node.output_names[i]] = op_node.outputs[i]


class PyTorchNode:
    r'''Base class for PyTorch scalar, tensors, and operators'''

    def __init__(self, graph):
        self.graph = graph
        self.input_names = []
        self.inputs = []
        self.output_names = []
        self.outputs = []

    def get_output_name(self, index):
        r'''Get the name of the output at the given index'''
        return self.output_names[index]

    def get_output(self, index):
        r'''Get the parsed output at the given index'''
        return self.outputs[index]

    def set_output(self, index, name, val):
        r'''Set the output at the given index with the specified name and value'''
        while len(self.output_names) <= index:
            self.output_names.append('')
        while len(self.outputs) <= index:
            self.outputs.append(None)
        self.output_names[index] = name
        self.outputs[index] = val


class PyTorchConstantTensor(PyTorchNode):
    r'''Base class for PyTorch input tensors and parameter tensors'''

    def __init__(self, name, arr, graph):
        super(PyTorchConstantTensor, self).__init__(graph)
        self.name = name
        self.arr = arr
        self.dtype = self.arr.dtype.name
        output = Variable(name=self.name, shape=self.shape,
                          dtype=self.dtype)
        self.set_output(0, name, output)

    @property
    def shape(self):
        r'''Get the shape of the tensor'''
        return list(self.arr.shape)


class PyTorchInput(PyTorchConstantTensor):
    r'''PyTorch input tensors'''

    def __init__(self, name, arr, graph):
        super(PyTorchInput, self).__init__(name, arr, graph)
        self.kind = 'input'


class PyTorchParam(PyTorchConstantTensor):
    r'''PyTorch parameter tensors'''

    def __init__(self, name, arr, graph):
        super(PyTorchParam, self).__init__(name, arr, graph)
        self.kind = 'param'


class PyTorchOutput(PyTorchNode):
    r'''PyTorch output tensors and scalars'''

    def __init__(self, name, val, graph):
        super(PyTorchOutput, self).__init__(graph)
        if isinstance(val, Symbol):
            self._shape = infer_shape(create(val))[1][0]
        self.set_output(0, name, val)

    @property
    def shape(self):
        r'''Get the shape of the output'''
        return self._shape[:]


class PyTorchOp(PyTorchNode):
    r'''Base class for PyTorch Prim and ATen operators'''

    def __init__(self, node, graph):
        super(PyTorchOp, self).__init__(graph)
        self.kind = node.kind()
        self.name = self.kind + '_' + str(len(self.graph.ops))
        self.input_names = []
        self.inputs = []
        for index, inp in enumerate(node.inputs()):
            input_name = inp.uniqueName()
            self.set_input(index, input_name, graph[input_name])
        for out in node.outputs():
            self.output_names.append(out.uniqueName())
        self._node = node

    def get_input_name(self, index):
        r'''Get the input name at the given index'''
        return self.input_names[index]

    def get_input(self, index):
        r'''Get the parsed input at the specified index'''
        return self.inputs[index]

    def set_input(self, index, name, val):
        r'''Set the input at the given index with the specified name and value'''
        while len(self.input_names) <= index:
            self.input_names.append('')
        while len(self.inputs) <= index:
            self.inputs.append(None)
        self.input_names[index] = name
        self.inputs[index] = val

    def set_output(self, index, name, val):
        node = PyTorchOutput(name, val, self.graph)
        super(PyTorchOp, self).set_output(index, name, node)
