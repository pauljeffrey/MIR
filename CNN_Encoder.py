import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
#from torchvision.models import DenseNet201

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.linear = self.__get_model()
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
            func = torch.nn.AdaptiveAvgPool2d((1,1))
        elif self.model_name == 'densenet201':
            densenet =models.densenet201(weights="IMAGENET1K_V1")  #models.densenet201(pretrained=self.pretrained)) #IMAGENET1K_V2"#weights=DenseNet201_Weights.DEFAULT) #
            #print("Densenet classifier in features: ", densenet.classifier.in_features)
            #print("Densenet classififer out features: ", densenet.classifier.out_features)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
      
        elif self.model_name == 'convnext_l':
            densenet =models.convnext_base(value = models.ConvNeXt_Large_Weights.IMAGENET1K_V1)  #models.densenet201(pretrained=self.pretrained)) #IMAGENET1K_V2"#weights=DenseNet201_Weights.DEFAULT) #
            #print("Densenet classifier in features: ", densenet.classifier.in_features)
            #print("Densenet classififer out features: ", densenet.classifier.out_features)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            #print(out_features)            
            
        elif self.model_name == 'convnext_t':
            densenet =models.ConvNeXt_Tiny_Weights(value=models.ConvNeXt_Tiny_Weights.DEFAULT)  #models.densenet201(pretrained=self.pretrained)) #IMAGENET1K_V2"#weights=DenseNet201_Weights.DEFAULT) #
            #print("Densenet classifier in features: ", densenet.classifier.in_features)
            #print("Densenet classififer out features: ", densenet.classifier.out_features)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        elif self.model_name == 'convnext_s':
            densenet =models.convnext_small(value=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)  #models.densenet201(pretrained=self.pretrained)) #IMAGENET1K_V2"#weights=DenseNet201_Weights.DEFAULT) #
            #print("Densenet classifier in features: ", densenet.classifier.in_features)
            #print("Densenet classififer out features: ", densenet.classifier.out_features)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        elif self.model_name == 'convnext_b':
            densenet =models.convnext_base(value= models.ConvNeXt_Base_Weights.IMAGENET1K_V1)  #models.densenet201(pretrained=self.pretrained)) #IMAGENET1K_V2"#weights=DenseNet201_Weights.DEFAULT) #
            #print("Densenet classifier in features: ", densenet.classifier.in_features)
            #print("Densenet classififer out features: ", densenet.classifier.out_features)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
        
        elif self.model_name == 'densenet121':
            densenet = models.densenet121(weights="IMAGENET1K_V1")
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        elif self.model_name == 'inception_v3':
            densenet = models.inception_v3(weights="IMAGENET1K_V1")
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        elif self.model_name == 'vgg19':
            densenet = models.vgg19(weights="IMAGENET1K_V1")
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        else:
            densenet = models.vgg16(weights="IMAGENET1K_V1")
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AdaptiveAvgPool2d((1,1))
            out_features = densenet.classifier.in_features
            
        linear = nn.Linear(in_features=out_features, out_features=out_features)
        # bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
        return model, out_features, func , linear  #, bn

    def forward(self, images):
        """
        :param images:
        :return:
        """
        visual_features = self.model(images)
        avg_features = self.avg_func(visual_features).squeeze()
        avg_features = self.activation(self.linear(avg_features)) #self.bn(self.linear(avg_features)))
        return visual_features, avg_features



class MLC(nn.Module):
    def __init__(self,
                 hidden_layer_size,
                 classes=156,
                 fc_in_features=2048,
                 
                 ):
        super(MLC, self).__init__()
        self.linear = nn.Linear(in_features=fc_in_features, out_features=hidden_layer_size)
        print("fc_in_features: ", fc_in_features)
        self.classifier = nn.Linear(in_features=hidden_layer_size, out_features=classes)        
        #self.softmax = nn.Softmax()
        self.sigmoid =nn.Sigmoid()
        #self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        avg_features = self.linear(avg_features)
        print("average features : ", avg_features.shape, torch.min(avg_features), torch.max(avg_features))
        if torch.any(torch.isnan(avg_features)) or torch.any(torch.isinf(avg_features)):
            print("avg features is nan")
        output = torch.nan_to_num(self.classifier(avg_features))
        print("output after classifying average: ", output.shape, torch.min(output), torch.max(output))
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print("average features is nan..")
        tags = self.sigmoid(output)
        #semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags #, semantic_features


class SemanticFeatureExtractor(nn.Module):
    def __init__(self,
                 classes =156,
                 semantic_features_dim=512,
                k = 10
                 ):
        super(SemanticFeatureExtractor, self).__init__()
        self.embed = nn.Embedding(classes, semantic_features_dim)
        self.k = k
        
    def forward(self, tags):
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return semantic_features
    
    
class Encoder(nn.Module):
    def __init__(self, model_name, pretrained, hidden_layer_size, classes, model_dict=None):
        super(Encoder, self).__init__()
        self.cnn_model = VisualFeatureExtractor(model_name, pretrained)
        fc_in_features = self.cnn_model.linear.out_features
        self.mlc = MLC(hidden_layer_size, classes, fc_in_features)
        self.flatten = nn.Flatten()
        if model_dict is not None:
            self.load_dict(model_dict)
        
    def load_dict(self,model_dict):
        if model_dict["cnn_model"] is not None:
            self.cnn_model.load_dict(model_dict["cnn_model"])
        if model_dict["mlc"] is not None:
            self.mlc.load_dict(model_dict["mlc"])
        
    def forward(self, images):
        visual_features, avg_features = self.cnn_model(images)
        tags = self.mlc(avg_features)#self.flatten()
        return visual_features, tags
    

if __name__ == '__main__':
    # model = VisualFeatureExtractor(pretrained=True)
    images = torch.randn((5,3, 224,224))
    # print(model.out_features)
    # print(model.avg_func)
    # print(model.linear)
    # #print(model.model.input)
    # print(model.model(images).shape)
    # print(model(images).shape)
    #self.model, self.out_features, self.avg_func, self.linear
    #model_name, pretrained, hidden_layer_size, classes, fc_in_features, model_dict=None
    model = Encoder("densenet201", True,128,156,)
    sem_ext = SemanticFeatureExtractor(156)
    
    output = model(images)
    print(output[0].shape, output[1].shape)
    print(sem_ext(output[1]).shape)