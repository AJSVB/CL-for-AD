from torch import nn
import torchvision.models as models
import torch.nn.functional as F


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


class Mean_shifted_AD_net(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        pretrained = hparams["pretrained"]
        print("pretrained" + str(pretrained))
        backbone = hparams["backbone"]
        if backbone == 152:
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n
