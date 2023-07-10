import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.bn, self.linear = self.__get_model()
        self.activation = nn.ReLU()

    def __get_model(self):
        model = None
        out_features = None
        func = None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2]
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        linear = nn.Linear(in_features=out_features, out_features=out_features)
        bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
        return model, out_features, func, bn, linear

    def forward(self, images):
        """
        :param images:
        :return:
        """
        visual_features = self.model(images)
        avg_features = self.avg_func(visual_features).squeeze()
        # avg_features = self.activation(self.bn(self.linear(avg_features)))
        return visual_features, avg_features


class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features
