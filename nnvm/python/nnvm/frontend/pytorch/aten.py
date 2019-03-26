'''This file contains one class per PyTorch ATen operator. For the full
list of operators, see
https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/native_functions.yaml
'''
import operator
from functools import reduce
import numpy as np
import tvm
from nnvm.symbol import Symbol
from .base import PyTorchOp, attr_2d, make_symbol


class ATenOp(PyTorchOp):
    r'''Base class for ATen operators'''

    def __init__(self, node, graph):
        super(ATenOp, self).__init__(node, graph)
        self.dtype = 'float32'


class Device(ATenOp):
    r'''aten::device operator'''

    def __init__(self, node, graph):
        super(Device, self).__init__(node, graph)
        self.set_output(0, self.get_output_name(0), None)


class AllSame(ATenOp):
    r'''Base class of aten::ones and aten::zeros'''

    def __init__(self, node, graph, val):
        super(AllSame, self).__init__(node, graph)
        val = float(val)
        shape = self.get_input(0).get_output(0)
        if not shape:
            self.set_output(0, self.get_output_name(0), val)
        else:
            attrs = {
                'shape': shape,
                'dtype': 'float32',
                'fill_value': val,
            }
            self.set_output(0, self.get_output_name(0),
                            make_symbol('full', **attrs))


class Ones(AllSame):
    r'''aten::ones operator'''
    def __init__(self, node, graph):
        super(Ones, self).__init__(node, graph, 1)


class Zeros(AllSame):
    r'''aten::zeros operator'''
    def __init__(self, node, graph):
        super(Zeros, self).__init__(node, graph, 0)


class HardTanh(ATenOp):
    r'''aten::hardtanh and aten::hardtanh_ operators'''

    def __init__(self, node, graph):
        super(HardTanh, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        attrs = {
            'a_min': self.get_input(1).get_output(0),
            'a_max': self.get_input(2).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('clip', *inputs, **attrs))


class Conv2D(ATenOp):
    r'''aten::_convolution operator'''

    def __init__(self, node, graph):
        super(Conv2D, self).__init__(node, graph)
        if self.get_input(6).get_output(0):
            topi_name = 'conv2d_transpose'
        else:
            topi_name = 'conv2d'
        if self.get_input(2).get_output(0) is None:
            data_index, weight_index = 0, 1
            weight_shape = self.get_input(weight_index).shape
            channels = weight_shape[0]
            use_bias = False
#            bias_name = self.name + '_bias'
#            bias = np.zeros([channels]).astype('float32')
#            self.graph.add_param(bias_name, bias)
            inputs = [
                self.get_input(data_index).get_output(0),
                self.get_input(weight_index).get_output(0),
#                self.graph[bias_name].get_output(0),
            ]
        else:
            use_bias = True
            data_index, weight_index, bias_index = 0, 1, 2
            weight_shape = self.get_input(weight_index).shape
            channels = weight_shape[0]
            inputs = [
                self.get_input(data_index).get_output(0),
                self.get_input(weight_index).get_output(0),
                self.get_input(bias_index).get_output(0),
            ]
        attrs = {
            'use_bias': use_bias,
            'channels': channels,
            'kernel_size': weight_shape[2:],
            'strides': self.get_input(3).get_output(0),
            'padding': self.get_input(4).get_output(0),
            'dilation': self.get_input(5).get_output(0),
            'groups': self.get_input(8).get_output(0),
            'kernel_layout': 'OIHW',
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol(topi_name, *inputs, **attrs))


class Threshold(ATenOp):
    r'''aten::threshold operator. Returns constant if input is less than or
    equal to threshold. Otherwise, returns input.'''

    def __init__(self, node, graph):
        super(Threshold, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        attrs = {
            'threshold': self.get_input(1).get_output(0),
            'constant': self.get_input(2).get_output(0),
        }
        if attrs['threshold'] != attrs['constant']:
            msg = 'For aten::threshold_, threshold != constant is not ' \
                  'implemented.'
            raise RuntimeError(msg)
        self.set_output(0, self.get_output_name(0),
                        make_symbol('relu', *inputs, **attrs))


class Pad(ATenOp):
    r'''aten::constant_pad_nd operator'''

    def __init__(self, node, graph):
        super(Pad, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        padding = self.get_input(1).get_output(0)
        attrs = {
            'pad_width': list(zip(padding, padding)),
            'pad_value': self.get_input(2).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('pad', *inputs, **attrs))


class BatchNorm(ATenOp):
    r'''aten::batch_norm operator'''

    def __init__(self, node, graph):
        super(BatchNorm, self).__init__(node, graph)
        self.topi_name = 'batch_norm'
        data = self.get_input(0).get_output(0)
        channels = self.get_input(0).shape[1]
        if any(self.get_input(i).get_output(0) is None for i in [1, 2]):
            mean_index, stdev_index = 3, 4
            scale = center = False
        else:
            gamma_index, beta_index, mean_index, stdev_index = 1, 2, 3, 4
            scale = center = True
            gamma = self.get_input(gamma_index).get_output(0)
            beta = self.get_input(beta_index).get_output(0)
        mean = self.get_input(mean_index).get_output(0)
        stdev = self.get_input(stdev_index).get_output(0)
        inputs = [data]
        if scale:
            inputs.append(gamma)
        else:
            gamma = np.ones([channels]).astype('float32')
            gamma_name = self.name + '_gamma'
            self.graph.add_param(gamma_name, gamma)
            inputs.append(self.graph[gamma_name].get_output(0))
        if center:
            inputs.append(beta)
        else:
            beta = np.zeros([channels]).astype('float32')
            beta_name = self.name + '_beta'
            self.graph.add_param(beta_name, beta)
            inputs.append(self.graph[beta_name].get_output(0))
        inputs.extend([mean, stdev])
        attrs = {
            'axis': 1,
            'epsilon': self.get_input(7).get_output(0),
            'center': center,
            'scale': scale,
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('batch_norm', *inputs, **attrs))


class Concatenate(ATenOp):
    r'''aten::cat operator'''

    def __init__(self, node, graph):
        super(Concatenate, self).__init__(node, graph)
        inputs = self.get_input(0).get_output(0)
        attrs = {
            'axis': self.get_input(1).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('concatenate', *inputs, **attrs))


class PermuteAxes(ATenOp):
    r'aten::t, aten::transpose, aten::permute operators'''

    def __init__(self, node, graph):
        super(PermuteAxes, self).__init__(node, graph)
        ndims = len(self.get_input(0).shape)
        axes = list(range(ndims))
        num_inputs = len(self.inputs)
        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(self.get_input(i).get_output(0)) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = self.get_input(1).get_output(0)
        attrs = {
            'axes': axes,
        }
        inputs = [self.get_input(0).get_output(0)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('transpose', *inputs, **attrs))


class Size(ATenOp):
    r'''aten::size operator'''

    def __init__(self, node, graph):
        super(Size, self).__init__(node, graph)
        axis = self.get_input(1).get_output(0)
        self.set_output(0, self.get_output_name(0),
                        self.get_input(0).shape[axis])


class View(ATenOp):
    r'''aten::view operator'''

    def __init__(self, node, graph):
        super(View, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        attrs = {
            'shape': self.get_input(1).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('reshape', *inputs, **attrs))


class Select(ATenOp):
    r'''aten::select operator'''

    def __init__(self, node, graph):
        super(Select, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        self._dim = self.get_input(1).get_output(0)
        index = self.get_input(2).get_output(0)
        end = self.get_input(0).shape[:]
        end[self._dim] = index + 1
        begin = [0] * len(end)
        begin[self._dim] = index
        self.attrs = {
            'begin': begin,
            'end': end,
            'stride': 1,
        }
        sym = make_symbol('strided_slice', *inputs, **self.attrs)
        inputs = [sym]
        attrs = {
            'axis': self._dim,
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('squeeze', *inputs, **attrs))

    @property
    def shape(self):
        r'''Get the shape'''
        if not hasattr(self, '_shape'):
            begin = np.array(self.attrs['begin']).astype(int)
            end = np.array(self.attrs['end']).astype(int)
            shape = (end - begin).tolist()
        return shape[:self._dim] + shape[self._dim + 1:]


class Copy(ATenOp):
    r'''aten::copy operator'''

    def __init__(self, node, graph):
        super(Copy, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('copy', *inputs))


class ReLU(ATenOp):
    r'''aten::relu and aten::relu_ operators'''

    def __init__(self, node, graph):
        super(ReLU, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('relu', *inputs))


class Softmax(ATenOp):
    r'''aten::softmax operator'''

    def __init__(self, node, graph):
        super(Softmax, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        axis = self.get_input(1).get_output(0)
        self.set_output(0, self.get_output_name(0),
                        make_symbol('softmax', *inputs, axis=axis))


class LogSoftmax(ATenOp):
    r'''aten::log_softmax operator'''

    def __init__(self, node, graph):
        super(LogSoftmax, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        axis = self.get_input(1).get_output(0)
        self.set_output(0, self.get_output_name(0),
                        make_symbol('log_softmax', *inputs, axis=axis))


class Sigmoid(ATenOp):
    r'''aten::sigmoid operator'''

    def __init__(self, node, graph):
        super(Sigmoid, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('sigmoid', *inputs))


class MatMul(ATenOp):
    r'''aten::matmul operator'''

    def __init__(self, node, graph):
        super(MatMul, self).__init__(node, graph)
        inputs = [self.get_input(i).get_output(0) for i in range(2)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('matmul', *inputs))


class Dense(ATenOp):
    r'''aten::addmm operator'''

    def __init__(self, node, graph):
        super(Dense, self).__init__(node, graph)
        data_index, weight_index, bias_index = 1, 2, 0
        data = self.get_input(data_index).get_output(0)
        weight = self.get_input(weight_index).get_output(0)
        bias = self.get_input(bias_index).get_output(0)
        units = self.get_input(weight_index).shape[1]
        attrs = {
            'units': units,
        }
        alpha = self.get_input(4).get_output(0)
        beta = self.get_input(3).get_output(0)
        data *= alpha
        weight *= beta
        weight = make_symbol('transpose', weight, axes=[1, 0])
        inputs = [data, weight, bias]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('dense', *inputs, **attrs))


class MaxPool2D(ATenOp):
    r'''aten::max_pool2d_with_indices operator'''

    def __init__(self, node, graph):
        super(MaxPool2D, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        if attr_2d(self.get_input(4).get_output(0), 1) != [1, 1]:
            raise RuntimeError('Only dilation = 1 supported')
        attrs = {
            'pool_size': attr_2d(self.get_input(1).get_output(0)),
            'strides': attr_2d(self.get_input(2).get_output(0), 1),
            'padding': attr_2d(self.get_input(3).get_output(0), 0),
            'ceil_mode': self.get_input(5).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('max_pool2d', *inputs, **attrs))


class AvgPool2D(ATenOp):
    r'''aten::avg_pool2d operator'''

    def __init__(self, node, graph):
        super(AvgPool2D, self).__init__(node, graph)
        self.topi_name = 'avg_pool2d'
        inputs = [self.get_input(0).get_output(0)]
        attrs = {
            'pool_size': attr_2d(self.get_input(1).get_output(0)),
            'strides': attr_2d(self.get_input(2).get_output(0), 1),
            'padding': attr_2d(self.get_input(3).get_output(0), 0),
            'ceil_mode': self.get_input(4).get_output(0),
            'count_include_pad': self.get_input(5).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('avg_pool2d', *inputs, **attrs))


class AdaptivePool2D(ATenOp):
    r'''Base class for adaptive pooling operators such as
    aten::adaptive_avg_pool2d and aten::adaptive_max_pool2d'''

    def __init__(self, node, graph, pool_type):
        super(AdaptivePool2D, self).__init__(node, graph)
        topi_name = 'adaptive_{}_pool2d'.format(pool_type)
        inputs = [self.get_input(0).get_output(0)]
        attrs = {
            'output_size': self.get_input(1).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol(topi_name, *inputs, **attrs))


class AdaptiveAvgPool2D(AdaptivePool2D):
    r'''aten::adaptive_avg_pool2d operator'''

    def __init__(self, node, graph):
        super(AdaptiveAvgPool2D, self).__init__(node, graph, 'avg')


class AdaptiveMaxPool2D(AdaptivePool2D):
    r'''aten::adaptive_max_pool2d operator'''

    def __init__(self, node, graph):
        super(AdaptiveMaxPool2D, self).__init__(node, graph, 'max')


class Dropout(ATenOp):
    r'''aten::dropout operator'''

    def __init__(self, node, graph):
        super(Dropout, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        if self.get_input(2).get_output(0):
            rate = 0
        else:
            rate = self.get_input(1).get_output(0)
        attrs = {
            'rate': rate,
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('dropout', *inputs, **attrs))


class Slice(ATenOp):
    r'''aten::slice operator'''

    def __init__(self, node, graph):
        super(Slice, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        end = self.get_input(0).shape[:]
        begin = [0] * len(end)
        dim = self.get_input(1).get_output(0)
        begin[dim] = self.get_input(2).get_output(0)
        end[dim] = min(end[dim], self.get_input(3).get_output(0))
        attrs = {
            'begin': begin,
            'end': end,
            'stride': self.get_input(4).get_output(0),
        }
        self.set_output(0, self.get_output_name(0),
                        make_symbol('strided_slice', *inputs, **attrs))


class BinaryOp(ATenOp):
    r'''Base class for binary operators such as aten::add and aten::mul'''

    def __init__(self, node, graph, operator_name):
        def prep(node):
            out = node.get_output(0)
            if isinstance(out, Symbol):
                return out
            if isinstance(out, tvm.nd.NDArray):
                out = out.asnumpy()
            return float(out)
        ATenOp.__init__(self, node, graph)
        linput, rinput = [prep(self.get_input(i)) for i in [0, 1]]
        if not all(isinstance(inp, Symbol) for inp in [linput, rinput]):
            self.set_output(0, self.get_output_name(0),
                            reduce(getattr(operator, operator_name), [linput, rinput]))
        else:
            topi_name = 'broadcast_' + operator_name
            self.set_output(0, self.get_output_name(0),
                            make_symbol(topi_name, linput, rinput))


class Subtract(BinaryOp):
    r'''aten::sub and aten::sub_ operators'''

    def __init__(self, node, graph):
        super(Subtract, self).__init__(node, graph, 'sub')


class Add(BinaryOp):
    r'''aten::add and aten::add_ operators'''

    def __init__(self, node, graph):
        super(Add, self).__init__(node, graph, 'add')


class Multiply(BinaryOp):
    r'''aten::mul and aten::mul_ operators'''

    def __init__(self, node, graph):
        super(Multiply, self).__init__(node, graph, 'mul')


class Divide(BinaryOp):
    r'''aten::div and aten::div_ operators'''

    def __init__(self, node, graph):
        super(Divide, self).__init__(node, graph, 'div')


class Unsqueeze(ATenOp):
    r'''aten::unsqueeze operator'''

    def __init__(self, node, graph):
        super(Unsqueeze, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        axis = self.get_input(1).get_output(0)
        self.set_output(0, self.get_output_name(0),
                        make_symbol('expand_dims', *inputs, axis=axis))


class Expand(ATenOp):
    r'''aten::expand operator'''

    def __init__(self, node, graph):
        super(Expand, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        shape = self.get_input(0).shape
        ndims = len(shape)
        sizes = self.get_input(1).get_output(0)
        self._shape = [max(shape[i], sizes[i]) for i in range(ndims)]
        out = self.get_input(0).get_output(0)
        out = self.get_input(0).get_output(0)
        for i in range(ndims):
            if sizes[i] in {-1, shape[i]}:
                continue
            inputs = [out] * sizes[i]
            out = make_symbol('concatenate', *inputs, axis=i)
        self.set_output(0, self.get_output_name(0), out)


class To(ATenOp):
    r'''aten::to operator'''

    def __init__(self, node, graph):
        super(To, self).__init__(node, graph)
        self.set_output(0, self.get_output_name(0),
                        self.get_input(0).get_output(0))


class Pow(ATenOp):
    r'''aten::pow operator'''

    def __init__(self, node, graph):
        super(Pow, self).__init__(node, graph)
        val = self.get_input(1).get_output(0)
        self.set_output(0, self.get_output_name(0),
                        self.get_input(0).get_output(0) ** val)


class Chunk(ATenOp):
    r'''aten::chunk operator'''

    def __init__(self, node, graph):
        super(Chunk, self).__init__(node, graph)
        num_chunks = self.get_input(1).get_output(0)
        axis = self.get_input(2).get_output(0)
        shape = self.get_input(0).shape
        dim = int(shape[axis])
        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)
        chunks = []
        for i in range(0, dim, unif_size):
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = i
            end[axis] = i + unif_size
            attrs = {
                'begin': begin,
                'end': end,
                'stride': [1] * len(shape),
            }
            chunk = make_symbol('strided_slice',
                                self.get_input(0).get_output(0),
                                **attrs)
            chunks.append(chunk)
        if dim % num_chunks:
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = unif_size * (num_chunks - 1)
            end[axis] = dim
            attrs = {
                'begin': begin,
                'end': end,
                'stride': [1] * len(shape),
            }
            chunk = make_symbol('strided_slice',
                                self.get_input(0).get_output(0),
                                **attrs)
            chunks.append(chunk)
        self.set_output(0, self.get_output_name(0), chunks)


class Reduce(ATenOp):
    r'''Base class for reduce operations such as aten::max, aten::sum, and aten::prod'''

    def __init__(self, node, graph, topi_name):
        super(Reduce, self).__init__(node, graph)
        if len(self.inputs) > 1:
            inputs = [self.get_input(0).get_output(0)]
            axis = self.get_input(1).get_output(0)
        else:
            inputs = [self.get_input(0).get_output(0)]
            axis = list(range(len(self.inputs[0].shape)))
        self.set_output(0, self.get_output_name(0),
                        make_symbol(topi_name, *inputs, axis=axis))


class Max(BinaryOp, Reduce, ATenOp):
    r'''Converts all aten::max operations, including both the binary op and the reduce op'''

    def __init__(self, node, graph):
        def is_binary_op_arg(node):
            out = node.get_output(0)
            return isinstance(out, (Symbol, tvm.nd.NDArray))
        ATenOp.__init__(self, node, graph)
        if len(self.inputs) > 1:
            if all(is_binary_op_arg(self.get_input(i)) for i in [0, 1]):
                BinaryOp.__init__(self, node, graph, 'max')
                return
        Reduce.__init__(self, node, graph, 'max')


class Sum(Reduce):
    r'''Sum over all elements of the input tensor or along specified axes'''

    def __init__(self, node, graph):
        super(Sum, self).__init__(node, graph, 'sum')


class Min(Reduce):
    r'''Compute the min over all elements of the input tensor or along specified axes'''

    def __init__(self, node, graph):
        super(Min, self).__init__(node, graph, 'min')


class Prod(Reduce):
    r'''Compute the product of all elements of the input tensor or along specified axes'''

    def __init__(self, node, graph):
        super(Prod, self).__init__(node, graph, 'prod')


class Mean(Reduce):
    r'''Compute the mean of all elements of the input tensor or along specified axes'''
    def __init__(self, node, graph):
        super(Mean, self).__init__(node, graph, 'mean')


class Sqrt(ATenOp):
    r'''Compute the elementwise square root'''

    def __init__(self, node, graph):
        super(Sqrt, self).__init__(node, graph)
        inputs = [self.get_input(0).get_output(0)]
        self.set_output(0, self.get_output_name(0),
                        make_symbol('sqrt', *inputs))


ATEN_MAP = {
    'device': Device,
    'ones': Ones,
    'zeros': Zeros,
    'hardtanh': HardTanh,
    'hardtanh_': HardTanh,
    '_convolution': Conv2D,
    'threshold': Threshold,
    'threshold_': Threshold,
    'constant_pad_nd': Pad,
    'contiguous': Copy,
    'batch_norm': BatchNorm,
    'cat': Concatenate,
    't': PermuteAxes,
    'transpose': PermuteAxes,
    'transpose_': PermuteAxes,
    'permute': PermuteAxes,
    'size': Size,
    'view': View,
    'select': Select,
    'clone': Copy,
    'relu': ReLU,
    'relu_': ReLU,
    'softmax': Softmax,
    'log_softmax': LogSoftmax,
    'sigmoid': Sigmoid,
    'addmm': Dense,
    'matmul': MatMul,
    'max_pool2d': MaxPool2D,
    'max_pool2d_with_indices': MaxPool2D,
    'avg_pool2d': AvgPool2D,
    'adaptive_max_pool2d': AdaptiveMaxPool2D,
    'adaptive_avg_pool2d': AdaptiveAvgPool2D,
    'dropout': Dropout,
    'slice': Slice,
    'sub': Subtract,
    'sub_': Subtract,
    'add': Add,
    'add_': Add,
    'mul': Multiply,
    'mul_': Multiply,
    'div': Divide,
    'div_': Divide,
    'unsqueeze': Unsqueeze,
    'expand': Expand,
    'to': To,
    'pow': Pow,
    'chunk': Chunk,
    'max': Max,
    'sum': Sum,
    'min': Min,
    'prod': Prod,
    'mean': Mean,
    'sqrt': Sqrt,
}
