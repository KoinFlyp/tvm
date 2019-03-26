r'''Unit tests for various models and operators'''
from time import time
import os
import sys
from tempfile import TemporaryDirectory
from scipy.stats import t as tdistr
import numpy as np
import torch
import tvm
import nnvm
import torchvision
import single_op
from mnist import Net
import mobilenet


sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    TARGET = tvm.target.cuda()
    CTX = tvm.gpu()
else:
    TARGET = 'llvm -mcpu=skylake-avx512'
    CTX = tvm.cpu()


def _vectorize(ten):
    return ten.reshape(-1)


def atol(tru, est):
    def _atol_elt(tru, est):
        return abs(tru - est)
    tru = _vectorize(tru)
    est = _vectorize(est)
    return max([_atol_elt(x, y) for x, y in zip(tru, est)])


def rtol(tru, est):
    def _rtol_elt(tru, est):
        return abs(tru - est) / min(abs(tru), abs(est))
    tru = _vectorize(tru)
    est = _vectorize(est)
    return max([_rtol_elt(x, y) for x, y in zip(tru, est)])


def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def load_torchvision(model_name):
    r'''Given a model name, returns a Torchvision model in eval mode as well
    as an example input.'''
    if model_name.startswith('inception'):
        height = width = 299
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        height = width = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    input_shape = [1, 3, height, width]
    input_data = torch.randn(input_shape).float()
    for channel in range(3):
        input_data[:, channel] -= mean[channel]
        input_data[:, channel] /= std[channel]
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.float().eval()
    return model, input_data


def load_pretrainedmodels(model_name):
    r'''Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input.'''
    import pretrainedmodels # https://github.com/Cadene/pretrained-models.pytorch
    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, input_data


def load_mobilenet(model_name):
    r'''Given a model name, returns a MobileNet model in eval mode as well as
    an example input.'''
    class_name = 'MobileNet' + model_name[-2:].capitalize()
    model = getattr(mobilenet, class_name)().float().eval()
    input_shape = [1, 3, 224, 224]
    input_data = torch.rand(input_shape).float() * 256
    imagenet_mean = [123., 117., 104.]
    imagenet_stdev = [58.395, 57.12, 57.375]
    for channel in range(3):
        input_data[:, channel] -= imagenet_mean[channel]
        input_data[:, channel] /= imagenet_stdev[channel]
    return model, input_data


def load_mnist():
    r'''Returns a MNIST model in eval mode as well as an example input.'''
    model = Net()
    input_shape = [1, 1, 28, 28]
    input_data = torch.rand(input_shape).float() * 256
    return model, input_data


def load_single_op(model_name):
    r'''Given a model name, returns a single-operator model in eval
    mode as well as an example input.'''
    model = getattr(single_op, model_name)().float().eval()
    input_shape = [1, 3, 224, 224]
    input_data = torch.rand(input_shape).float()
    return model, input_data


def load_fastai():
    r'''Returns a FastAI model as well as an example input.'''
    model = torch.jit.load('fastai.pth', map_location='cpu')
    input_shape = [1, 3, 224, 224]
    input_data = torch.rand(input_shape).float()
    return model, input_data


def load_sfd():
    from net_s3fd import s3fd
    model = s3fd()
    input_shape = [1, 3, 512, 512]
    input_data = torch.rand(input_shape).float()
    return model, input_data


def load_model(model_name):
    r'''Given a model name, returns a model as well as an example input.'''
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    if model_name.startswith('mobilenet'):
        return load_mobilenet(model_name)
    if model_name == 'mnist':
        return load_mnist()
    if hasattr(single_op, model_name):
        return load_single_op(model_name)
    if model_name == 'fastai':
        return load_fastai()
    if model_name == 'sfd':
        return load_sfd()
    try:
        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Please install pretrainedmodels.pytorch')
    raise RuntimeError('Model not supported')


def confidence_interval(mean, stdev, count, alpha=.01):
    r'''Returns the lower and upper bounds of the confidence interval of a random
    variable. Confidence is 1 - alpha (default confidence is 99%).'''
    stdval = tdistr.ppf(1 - alpha / 2, count - 1)
    lower, upper = mean + np.array([-1, 1]) * stdval * stdev / np.sqrt(count)
    return lower, upper

def measure_latency(model, input_shapes, output_shapes, thresh, dryruns=40):
    r'''Compute the latency of the given model'''
    latencies = []
    count = 0
    while True:
        if isinstance(model, torch.nn.Module):
            input_data = [torch.rand(shape).float() for shape in input_shapes]
            if torch.cuda.is_available():
                input_data = list(map(lambda x: x.cuda(), input_data))
                model = model.cuda()
            t_start = time()
            model(*input_data)
            t_end = time()
            latencies.append(t_end - t_start)
        else:
            input_data = {}
            for i, shape in enumerate(input_shapes):
                name = 'input' + str(i)
                arr = np.random.random(shape).astype('float32')
                input_data[name] = tvm.nd.array(arr)
            t_start = time()
            model.set_input(**input_data)
            model.run()
            for i, shape in enumerate(output_shapes):
                arr = np.zeros(shape).astype('float32')
                model.get_output(i, tvm.nd.array(arr))
            t_end = time()
        count += 1
        if count < dryruns:
            continue
        latencies.append(t_end - t_start)
        mean = np.mean(latencies)
        stdev = np.std(latencies)
        sample_size = len(latencies)
        if sample_size > dryruns:
            lower, upper = confidence_interval(mean, stdev, sample_size)
            est = (upper + lower) / 2
            err = (upper - lower) / 2
            if err < thresh:
                return est
            print(f'Latency so far is {est:.3f} +/- {err:.3f} seconds.')

def verify_model(model_name):
    r'''Assert that the output of a compiled model matches with that of its
    baseline.'''
    baseline_model, baseline_input = load_model(model_name)
    if torch.cuda.is_available():
        baseline_model = baseline_model.cuda()
        baseline_input = baseline_input.cuda()
    baseline_outputs = baseline_model(baseline_input)
    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.detach().cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.detach().cpu().numpy(),)
    output_shapes = [out.shape for out in baseline_outputs]
    dtype = 'float32'
    input_name = 'input0'
    input_shapes = {input_name: list(baseline_input.shape)}
    baseline_model(baseline_input)
    trace = torch.jit.trace(baseline_model, baseline_input).float().eval()
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.pth')
        torch.jit.save(trace, path)
        sym, params = nnvm.frontend.from_pytorch(path, input_shapes)
    compiled_input = {input_name: tvm.nd.array(baseline_input.cpu().numpy())}
    graph, lib, params = nnvm.compiler.build(sym, TARGET, input_shapes,
                                             dtype='float32',
                                             params=params)
    compiled_model = tvm.contrib.graph_runtime.create(graph, lib, CTX)
    compiled_model.set_input(**params)
    compiled_model.set_input(**compiled_input)
    compiled_model.run()
    for i, baseline_output in enumerate(baseline_outputs):
        output_shape = baseline_output.shape
        compiled_output = compiled_model.get_output(
            i, tvm.nd.array(np.zeros(output_shape).astype(dtype), CTX)).asnumpy()
        assert_shapes_match(baseline_output, compiled_output)
        tvm.testing.assert_allclose(baseline_output, compiled_output,
                                    rtol=1e-5, atol=1e-5)
    thresh = 1e-2
    units = 1e3
    input_shapes = list(input_shapes.values())
    baseline_latency = measure_latency(baseline_model, input_shapes,
                                       output_shapes, thresh) * units
    compiled_latency = measure_latency(compiled_model, input_shapes,
                                       output_shapes, thresh) * units
    thresh = int(thresh * units)
    print(f'Baseline latency is {baseline_latency:.3f} +/- {thresh:d} ms.')
    print(f'Compiled latency is {compiled_latency:.3f} +/- {thresh:d} ms.')


def verify_ones1():
    verify_model('Ones1')

def verify_zeros1():
    verify_model('Zeros1')

def verify_add1():
    verify_model('Add1')

def verify_add2():
    verify_model('Add2')

def verify_add3():
    verify_model('Add3')

def verify_add4():
    verify_model('Add4')

def verify_add5():
    verify_model('Add5')

def verify_subtract1():
    verify_model('Subtract1')

def verify_subtract2():
    verify_model('Subtract2')

def verify_subtract3():
    verify_model('Subtract3')

def verify_subtract4():
    verify_model('Subtract4')

def verify_subtract5():
    verify_model('Subtract5')

def verify_multiply1():
    verify_model('Multiply1')

def verify_multiply2():
    verify_model('Multiply2')

def verify_multiply3():
    verify_model('Multiply3')

def verify_multiply4():
    verify_model('Multiply4')

def verify_multiply5():
    verify_model('Multiply5')

def verify_unsqueeze1():
    verify_model('Unsqueeze1')

def verify_concatenate1():
    verify_model('Concatenate1')

def verify_concatenate2():
    verify_model('Concatenate2')

def verify_relu1():
    verify_model('ReLU1')

def verify_adaptiveavgpool2d1():
    verify_model('AdaptiveAvgPool2D1')

def verify_adaptiveavgpool2d2():
    verify_model('AdaptiveAvgPool2d2')

def verify_adaptiveavgpool2d3():
    verify_model('AdaptiveAvgPool2d3')

def verify_maxpool2d1():
    verify_model('MaxPool2D1')

def verify_maxpool2d2():
    verify_model('MaxPool2D2')

def verify_maxpool2d3():
    verify_model('MaxPool2D3')

def verify_hardtanh1():
    verify_model('HardTanh1')

def verify_conv2d1():
    verify_model('Conv2D1')

def verify_threshold1():
    verify_model('Threshold1')

def verify_contiguous1():
    verify_model('Contiguous1')

def verify_batchnorm1():
    verify_model('BatchNorm1')

def verify_transpose1():
    verify_model('Transpose1')

def verify_transpose2():
    verify_model('Transpose2')

def verify_size1():
    verify_model('Size1')

def verify_view1():
    verify_model('View1')

def verify_view2():
    verify_model('View2')

def verify_select1():
    verify_model('Select1')

def verify_clone1():
    verify_model('Clone1')

def verify_logsoftmax1():
    verify_model('LogSoftmax1')

def verify_sigmoid1():
    verify_model('Sigmoid1')

def verify_dense1():
    verify_model('Dense1')

def verify_avgpool2d1():
    verify_model('AvgPool2D1')

def verify_dropout1():
    verify_model('Dropout1')

def verify_slice1():
    verify_model('Slice1')

def verify_slice2():
    verify_model('Slice2')

def verify_mean1():
    verify_model('Mean1')

def verify_expand1():
    verify_model('Expand1')

def verify_pow1():
    verify_model('Pow1')

def verify_chunk1():
    verify_model('Chunk1')

def verify_alexnet():
    verify_model('alexnet')

def verify_densenet121():
    verify_model('densenet121')

def verify_densenet161():
    verify_model('densenet161')

def verify_densenet169():
    verify_model('densenet169')

def verify_densenet201():
    verify_model('densenet201')

def verify_inception_v3():
    verify_model('inception_v3')

def verify_resnet101():
    verify_model('resnet101')

def verify_resnet152():
    verify_model('resnet152')

def verify_resnet18():
    verify_model('resnet18')

def verify_resnet34():
    verify_model('resnet34')

def verify_resnet50():
    verify_model('resnet50')

def verify_squeezenet1_0():
    verify_model('squeezenet1_0')

def verify_squeezenet1_1():
    verify_model('squeezenet1_1')

def verify_vgg11():
    verify_model('vgg11')

def verify_vgg11_bn():
    verify_model('vgg11_bn')

def verify_vgg13():
    verify_model('vgg13')

def verify_vgg13_bn():
    verify_model('vgg13_bn')

def verify_vgg16():
    verify_model('vgg16')

def verify_vgg16_bn():
    verify_model('vgg16_bn')

def verify_vgg19():
    verify_model('vgg19')

def verify_vgg19_bn():
    verify_model('vgg19_bn')

def verify_sfd():
    verify_model('sfd')


if __name__ == '__main__':
#    verify_sfd()
    verify_add1()
    verify_add2()
    verify_add3()
    verify_add4()
    verify_add5()
    verify_subtract1()
    verify_subtract2()
    verify_subtract3()
    verify_subtract4()
    verify_subtract5()
    verify_multiply1()
    verify_multiply2()
    verify_multiply3()
    verify_multiply4()
    verify_multiply5()
    verify_unsqueeze1()
    verify_concatenate1()
    verify_concatenate2()
    verify_relu1()
#    verify_adaptiveavgpool2d1()
#    verify_adaptiveavgpool2d2()
#    verify_adaptiveavgpool2d3()
    verify_maxpool2d1()
    verify_maxpool2d2()
    verify_maxpool2d3()
    verify_hardtanh1()
    verify_conv2d1()
    verify_threshold1()
    verify_contiguous1()
    verify_batchnorm1()
    verify_transpose1()
    verify_transpose2()
    verify_size1()
    verify_view1()
    verify_view2()
    verify_select1()
    verify_clone1()
    verify_logsoftmax1()
    verify_sigmoid1()
    verify_dense1()
    verify_avgpool2d1()
    verify_dropout1()
    verify_slice1()
    verify_slice2()
    verify_mean1()
    verify_expand1()
    verify_pow1()
    verify_chunk1()
    exit()
    verify_alexnet()
    verify_densenet121()
    verify_densenet161()
    verify_densenet169()
    verify_densenet201()
    verify_inception_v3()
    verify_resnet101()
    verify_resnet152()
    verify_resnet18()
    verify_resnet34()
    verify_resnet50()
    verify_squeezenet1_0()
    verify_squeezenet1_1()
    verify_vgg11()
    verify_vgg11_bn()
    verify_vgg13()
    verify_vgg13_bn()
    verify_vgg16()
    verify_vgg16_bn()
    verify_vgg19()
    verify_vgg19_bn()
