import torch
import torch.nn as nn
from torchvision.models import resnet34, vgg19
from models.MatchingNetwork import MatchingNetwork


class ExpanderLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ExpanderLayer, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class VICReg(nn.Module):
    def __init__(
        self,
        base_model: str = "CustomCNN",
        encoder_features: int = 1600,
        expander_features: int = 1024,
        pars_extra: dict = {}
    ):
        super(VICReg, self).__init__()
        self.base_model = base_model
        self.encoder_features = encoder_features
        self.expander_features = expander_features

        if self.base_model == "CustomCNN":
            self.encoder = CustomCNN(encoder_features=encoder_features)
        elif self.base_model == "Resnet34":
            self.encoder = ResnetEncoderVICReg(encoder_features=encoder_features)
        elif self.base_model == "Vgg19":
            self.encoder = VggEncoderVICReg(encoder_features=encoder_features)
        elif self.base_model == "MatchingNetwork":
            self.encoder = CustomCNN(encoder_features=encoder_features)
            
            # self.encoder = CustomMatchingNetwork(layer_size = pars_extra["layer_size"], num_channels=pars_extra["num_channels"],
                                        #         nClasses= pars_extra["nClasses"], image_size = pars_extra["image_size"]
                                        #  )

        self.expander = nn.Sequential(
            ExpanderLayer(encoder_features, expander_features),
            ExpanderLayer(expander_features, expander_features),
            nn.Linear(expander_features, expander_features),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.expander(x)
        return x

    def save(self, path):
        torch.save(
            {
                "encoder_features": self.encoder_features,
                "expander_features": self.expander_features,
                "encoder_state_dict": self.encoder.state_dict(),
            },
            path,
        )


# --------------------------------------------------------------------------- CUSTOM CNN


class CustomCNN(nn.Module):
    def __init__(self, encoder_features: int = 6400):
        super(CustomCNN, self).__init__()
        # Nuñez-Alcover, A., León, P. J., & Calvo-Zaragoza, J.
        # Glyph and position classification of music symbols in early music manuscripts
        layers = [
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(),
        ]
        if encoder_features != 6400:
            layers.append(nn.Linear(6400, encoder_features))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)


class SupervisedClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super(SupervisedClassifier, self).__init__()
        # Nuñez-Alcover, A., León, P. J., & Calvo-Zaragoza, J.
        # Glyph and position classification of music symbols in early music manuscripts
        self.encoder = CustomCNN()
        self.decoder = nn.Sequential(
            nn.Linear(6400, 256), nn.Dropout(0.25), nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --------------------------------------------------------------------------- RESNET34


class ResnetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(ResnetEncoder, self).__init__()
        resnet = resnet34(pretrained=pretrained)
        resnet_modules = list(resnet.children())
        self.backbone = nn.Sequential(*resnet_modules[:-1], nn.Flatten())

    def forward(self, x):
        return self.backbone(x)


class ResnetEncoderVICReg(nn.Module):
    def __init__(self, encoder_features: int = 1600, pretrained: bool = False):
        super(ResnetEncoderVICReg, self).__init__()
        resnet = resnet34(pretrained=pretrained)
        resnet_modules = list(resnet.children())
        self.backbone = nn.Sequential(*resnet_modules[:-1], nn.Flatten())
        self.out = nn.Linear(512, encoder_features)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)


class ResnetClassifier(nn.Module):
    def __init__(self, num_labels, pretrained: bool = True):
        super(ResnetClassifier, self).__init__()
        self.resnet = resnet34(pretrained=pretrained)
        resnet_modules = list(self.resnet.children())
        self.backbone = nn.Sequential(*resnet_modules[:-1], nn.Flatten())
        self.decoder = nn.Sequential(
            nn.Linear(512, 256), nn.Dropout(0.25), nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.decoder(x)


# --------------------------------------------------------------------------- VGG19


class VggEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(VggEncoder, self).__init__()
        vgg = vgg19(pretrained=pretrained)
        vgg_modules = list(vgg.children())
        self.backbone = nn.Sequential(*vgg_modules[:-1], nn.Flatten())

    def forward(self, x):
        return self.backbone(x)


class VggEncoderVICReg(nn.Module):
    def __init__(self, encoder_features: int = 1600, pretrained: bool = False):
        super(VggEncoderVICReg, self).__init__()
        vgg = vgg19(pretrained=pretrained)
        vgg_modules = list(vgg.children())
        self.backbone = nn.Sequential(*vgg_modules[:-1], nn.Flatten())
        self.out = nn.Linear(25088, encoder_features)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)


class VggClassifier(nn.Module):
    def __init__(self, num_labels, pretrained: bool = True):
        super(VggClassifier, self).__init__()
        self.vgg = vgg19(pretrained=pretrained)
        vgg_modules = list(self.vgg.children())
        self.backbone = nn.Sequential(*vgg_modules[:-1], nn.Flatten())
        self.decoder = nn.Sequential(
            nn.Linear(25088, 256), nn.Dropout(0.25), nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.decoder(x)



# --------------------------------------------------------------------------- CustomMatchingNetwork

from models.Classifier import Classifier
## Use only of Classifier
class CustomMatchingNetwork(nn.Module):
    def __init__(self, layer_size = 64, nClasses = 0, num_channels = 1, image_size = 28, encoder_features: int = 1600 ):

        super(ResnetEncoderVICReg, self).__init__()
        resnet = resnet34(pretrained=True)
        resnet_modules = list(resnet.children())
        self.backbone = nn.Sequential(*resnet_modules[:-1], nn.Flatten())
        self.lastlayer = nn.Linear(512, encoder_features)

    def forward(self, x):
        x = self.backbone(x)
        return self.lastlayer(x)
        
        
        
        # super(CustomMatchingNetwork, self).__init__()
        # classif = Classifier(layer_size = layer_size, num_channels=num_channels, nClasses= nClasses, image_size = image_size )
        
        # self.layer1 = classif.layer1
        # self.layer2 = classif.layer2
        # self.layer3 = classif.layer3
        # self.layer4 = classif.layer4
        
        # # self.lastlayer = nn.Linear(6400, layer_size)
        # self.lastlayer = nn.Linear(256, 1600)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.lastlayer(x)

# class CustomMatchingNetworkEncoderVICReg(nn.Module):
#     def __init__(self, encoder_features, checkpoint, params_extra):
#         super(CustomMatchingNetworkEncoderVICReg, self).__init__()
#         self.model = VICReg(
#             base_model="MatchingNetwork",
#             pars_extra=params_extra,
#         )
#         # vgg_modules = list(model.children())
#         # self.backbone = nn.Sequential(*vgg_modules[:-1], nn.Flatten())
#         # self.out = nn.Linear(25088, checkpoint["encoder_features"])

#     def forward(self, x):
#         return self.model(x)
#         x = self.backbone(x)
#         return self.out(x)



# class SupervisedClassifier(nn.Module):
#     def __init__(self, num_labels: int):
#         super(SupervisedClassifier, self).__init__()
#         # Nuñez-Alcover, A., León, P. J., & Calvo-Zaragoza, J.
#         # Glyph and position classification of music symbols in early music manuscripts
#         self.encoder = CustomCNN()
#         self.decoder = nn.Sequential(
#             nn.Linear(6400, 256), nn.Dropout(0.25), nn.Linear(256, num_labels)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x





