import timm
import torch
import torch.nn as nn
from torch.utils import model_zoo
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.resnet import BasicBlock, Bottleneck

# --- ResNet Implementation ---
# This section contains a standard ResNet implementation used by the model builder.

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class ResNet(nn.Module):
    def __init__(self, block, layers, classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.class_classifier(x)

def resnet18_custom(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50_custom(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

# --- Other Model Architectures from your file ---

def resnet50_no_layer4(pretrained=True):
    class ResNetNoLayer4(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            resnet = resnet50_custom(pretrained=pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.embed_dim = 1024
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    return ResNetNoLayer4(pretrained=pretrained)

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.class_classifier  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch", "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch", "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def resnet50_lunit(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    model.embed_dim = 2048
    return model

def vit_small_lunit(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    model.embed_dim = 384
    return model

# --- Custom Model Wrapper ---

class CustomModel(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, cfg.n_class)

    def forward(self, image, return_feature=False):
        image_features = self.encoder(image)
        logits = self.head(image_features)
        if return_feature:
            return logits, image_features
        return logits

# --- Model Builder Function (Corrected) ---

def build_model(cfg):
    encoder = None  # Initialize encoder to avoid UnboundLocalError

    # --- CORRECTED LOGIC: Check pretrain and backbone separately ---
    if cfg.pretrain == 'natural_supervised':
        if cfg.backbone == 'Resnet50':
            print("Building ImageNet-pretrained ResNet-50 encoder...")
            if hasattr(cfg, 'remove_layer4') and cfg.remove_layer4:
                encoder = resnet50_no_layer4(pretrained=True)
            else:
                encoder = timm.create_model('resnet50', pretrained=True)
                encoder.fc = nn.Identity()
                encoder.embed_dim = encoder.num_features
        
        elif cfg.backbone == 'Resnet18':
            print("Building ImageNet-pretrained ResNet-18 encoder...")
            encoder = resnet18_custom(pretrained=True)
            encoder.class_classifier = nn.Identity()
            encoder.embed_dim = 512 # ResNet-18 feature dimension
        
        elif cfg.backbone == 'ViT-B/16':
            print("Building ImageNet-pretrained ViT-Base/16 encoder...")
            encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
            encoder.head = nn.Identity()
            encoder.embed_dim = encoder.num_features

    elif cfg.pretrain == 'natural_ssl':
        if cfg.backbone == 'Resnet50':
            print("Building DINO-pretrained ResNet-50 encoder...")
            encoder = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            encoder.embed_dim = encoder.fc.in_features
        elif cfg.backbone == 'ViT-S/16':
            print("Building DINO-pretrained ViT-Small/16 encoder...")
            encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            encoder.embed_dim = encoder.head.in_features

    elif cfg.pretrain == 'medical_ssl':
        if cfg.backbone == 'Resnet50':
            print("Building Lunit-SSL-pretrained ResNet-50 encoder...")
            encoder = resnet50_lunit(pretrained=True, progress=False, key="BT")
        elif cfg.backbone == 'ViT-S/16':
            print("Building Lunit-SSL-pretrained ViT-Small/16 encoder...")
            encoder = vit_small_lunit(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
    
    elif cfg.pretrain == 'tailored_sl':
        print("Building Tailored-SL-pretrained ViT-Small/16 encoder...")
        encoder = vit_small_lunit(pretrained=True, progress=False, key="DINO_p16", patch_size=16)

    if encoder is None:
        raise NotImplementedError(f"Model construction not implemented for pretrain='{cfg.pretrain}' and backbone='{cfg.backbone}'")

    return CustomModel(cfg, encoder)
