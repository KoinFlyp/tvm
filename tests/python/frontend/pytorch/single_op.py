r'''Models consisting of single operators'''
import torch
from torch.nn import Module


class Add1(Module):
    def forward(self, *args):
        return args[0] + args[0]

class Add2(Module):
    def forward(self, *args):
        return args[0] + 1

class Add3(Module):
    def forward(self, *args):
        ones = torch.ones([1, 3, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] + ones

class Add4(Module):
    def forward(self, *args):
        ones = torch.ones([1, 1, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] + ones

class Add5(Module):
    def forward(self, *args):
        ones = torch.ones([])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] + ones

class Subtract1(Module):
    def forward(self, *args):
        return args[0] - args[0]

class Subtract2(Module):
    def forward(self, *args):
        return args[0] - 1

class Subtract3(Module):
    def forward(self, *args):
        ones = torch.ones([1, 3, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] - ones

class Subtract4(Module):
    def forward(self, *args):
        ones = torch.ones([1, 1, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] - ones

class Subtract5(Module):
    def forward(self, *args):
        ones = torch.ones([])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] - ones

class Multiply1(Module):
    def forward(self, *args):
        return args[0] * args[0]

class Multiply2(Module):
    def forward(self, *args):
        return args[0] * 1

class Multiply3(Module):
    def forward(self, *args):
        ones = torch.ones([1, 3, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] * ones

class Multiply4(Module):
    def forward(self, *args):
        ones = torch.ones([1, 1, 224, 224])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] * ones

class Multiply5(Module):
    def forward(self, *args):
        ones = torch.ones([])
        if torch.cuda.is_available():
            ones = ones.cuda()
        return args[0] * ones

class Unsqueeze1(Module):
    def forward(self, *args):
        return args[0].unsqueeze(2)

class Concatenate1(Module):
    def forward(self, *args):
        return torch.cat([args[0][:, 0].unsqueeze(1), args[0][:, 1].unsqueeze(1)], 1)

class Concatenate2(Module):
    def forward(self, *args):
        a = (args[0][:, :, 0] + 2) * 7
        b = (args[0][:, :, 1] + 3) * 11
        c = (args[0][:, :, 2] + 5) * 13
        return torch.cat([t.unsqueeze(2) for t in [a, b, c]], 2)

class ReLU1(Module):
    def forward(self, *args):
        return torch.nn.ReLU()(args[0])

class AdaptiveAvgPool2D1(Module):
    def forward(self, *args):
        return torch.nn.AdaptiveAvgPool2d([1, 1])(args[0])

class AdaptiveAvgPool2D2(Module):
    def forward(self, *args):
        return torch.nn.AdaptiveAvgPool2d([100, 100])(args[0])

class AdaptiveAvgPool2D3(Module):
    def forward(self, *args):
        return torch.nn.AdaptiveAvgPool2d([224, 224])(args[0])

class MaxPool2D1(Module):
    def forward(self, *args):
        return torch.nn.MaxPool2d(kernel_size=[1, 1])(args[0])

class MaxPool2D2(Module):
    def forward(self, *args):
        return torch.nn.MaxPool2d(kernel_size=[100, 100])(args[0])

class MaxPool2D3(Module):
    def forward(self, *args):
        return torch.nn.MaxPool2d(kernel_size=[224, 224])(args[0])

class HardTanh1(Module):
    def forward(self, *args):
        return torch.nn.Hardtanh()(args[0])

class Conv2D1(Module):

    def __init__(self):
        super(Conv2D1, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7)

    def forward(self, *args):
        return self.conv(args[0])

class Threshold1(Module):
    def forward(self, *args):
        return torch.nn.Threshold(0, 0)(args[0])

class Pad1(Module):
    def forward(self, *args):
        return torch.ConstantPad2d(3)(args[0])

class Contiguous1(Module):
    def forward(self, *args):
        return args[0].contiguous()

class BatchNorm1(Module):
    def __init__(self):
        super(BatchNorm1, self).__init__()
        self.batch_norm = torch.nn.BatchNorm2d(3)
    def forward(self, *args):
        return self.batch_norm(args[0])

class Transpose1(Module):
    def forward(self, *args):
        return args[0].transpose(2, 3)

class Transpose2(Module):
    def forward(self, *args):
        return args[0].transpose(-2, -1)

class Transpose3(Module):
    def forward(self, *args):
        return args[0].t()

class Size1(Module):
    def forward(self, *args):
        return args[0].size(0) * args[0]

class View1(Module):
    def forward(self, *args):
        return args[0].view((1, 3 * 224 * 224))

class View2(Module):
    def forward(self, *args):
        return args[0].view(args[0].shape[0], -1)

class Select1(Module):
    def forward(self, *args):
        return args[0].select(1, 1)

class Clone1(Module):
    def forward(self, *args):
        return args[0].clone()

class LogSoftmax1(Module):
    def forward(self, *args):
        return torch.nn.LogSoftmax(dim=1)(args[0][0, 0])

class Sigmoid1(Module):
    def forward(self, *args):
        return torch.nn.Sigmoid()(args[0])

class Dense1(Module):
    def __init__(self):
        super(Dense1, self).__init__()
        self.linear = torch.nn.Linear(224, 7)
    def forward(self, *args):
        return self.linear(args[0][0, 0])

class AvgPool2D1(Module):
    def forward(self, *args):
        return torch.nn.AvgPool2d(kernel_size=[100, 100])(args[0])

class Dropout1(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0][0, 0], 0.5, False)

class Slice1(Module):
    def forward(self, *args):
        return args[0][:, :, :, :3]

class Slice2(Module):
    def forward(self, *args):
        return args[0][0, :, :, :]

class Mean1(Module):
    def forward(self, *args):
        return args[0].mean(2)

class Expand1(Module):
    def forward(self, *args):
        return args[0].expand((3, -1, -1, -1))

class Pow1(Module):
    def forward(self, *args):
        return args[0] ** 2

class Chunk1(Module):
    def forward(self, *args):
        chunks = args[0].chunk(7, 2)
        return torch.cat(chunks, 2)
