import torch
import torch.nn as nn

from torch.autograd import Function, Variable
from utils.utils import sigmoid

class MultibranchLeNet(nn.Module):

    def __init__(self):
        # Construct nn.Module superclass from the derived classs MultibranchLeNet
        super(MultibranchLeNet, self).__init__()
        # Construct MultibranchLeNet architecture
        self.conv1 = nn.Sequential()
        self.conv1.add_module('c1_conv', nn.Conv2d(3, 32, kernel_size=5))
        self.conv1.add_module('c1_relu', nn.ReLU(True))
        self.conv1.add_module('c1_pool', nn.MaxPool2d(2))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('c2_conv', nn.Conv2d(32, 48, kernel_size=5))
        self.conv2.add_module('c2_relu', nn.ReLU(True))
        self.conv2.add_module('c2_pool', nn.MaxPool2d(2))

        self.g_conv1_d = nn.Parameter(torch.tensor(1.09))
        self.g_conv1_t = nn.Parameter(torch.tensor(1.09))
        self.g_conv2_d = nn.Parameter(torch.tensor(1.09))
        self.g_conv2_t = nn.Parameter(torch.tensor(1.09))

        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('f_fc1', nn.Linear(48 * 4 * 4, 100))
        self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu1', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc2', nn.Linear(100, 100))
        self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu2', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc3', nn.Linear(100, 10))
        self.feature_classifier.add_module('f_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(48 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input, lamda, plasticity):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        out1 = sigmoid(self.g_conv1_d, plasticity) * self.conv1(input) + \
               sigmoid(self.g_conv1_t, plasticity) * self.conv1(input)
        feature = sigmoid(self.g_conv2_d, plasticity) * self.conv2(out1) + \
                  sigmoid(self.g_conv2_t, plasticity) * self.conv2(out1)
        feature = feature.view(-1, 48 * 4 * 4)
        class_prediction = self.feature_classifier(feature)
        reverse_feature = ReverseLayer.apply(feature, lamda)
        domain_prediction = self.domain_classifier(reverse_feature)

        return class_prediction, domain_prediction


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None
