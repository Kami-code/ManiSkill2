import torch.nn as nn
import torch
from icecream import ic
def load_r3m(modelid):
    import os
    from os.path import expanduser
    import omegaconf
    import hydra
    import gdown
    import torch
    import copy
    VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight",
                  "l2dist", "bs"]
    device = "cuda:0"
    def cleanup_config(cfg):
        config = copy.deepcopy(cfg)
        keys = config.agent.keys()
        for key in list(keys):
            if key not in VALID_ARGS:
                del config.agent[key]
        config.agent["_target_"] = "r3m.R3M"
        config["device"] = device

        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        config.agent["langweight"] = 0
        return config.agent

    def remove_language_head(state_dict):
        keys = state_dict.keys()
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        for key in list(keys):
            if ("lang_enc" in key) or ("lang_rew" in key):
                del state_dict[key]
        return state_dict
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "res50":
        foldername = "r3m_50"
        modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
        configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    elif modelid == "res18":
        foldername = "r3m_18"
        modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
        configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])
    rep.load_state_dict(r3m_state_dict)
    return rep

class ResNet18(nn.Module):
    def __init__(self, output_channel, pretrain) -> None:
        super(ResNet18, self).__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        if not pretrain:
            self.vision_extractor = resnet18()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == 'IMAGENET1K':
            self.vision_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.vision_extractor.eval()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == 'R3M':
            self.vision_extractor = load_r3m('res18')
            self.vision_extractor.eval()
            self.fc = nn.Linear(512, output_channel).to('cuda')
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: B x 224 x 224 x 3
        # ic(x.shape)
        # x = torch.permute(x, (0, 3, 1, 2))  # x: B x 3 x 224 x 224
        out = self.vision_extractor(x[:, 0:3, :, :])
        out = self.fc(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, output_channel, pretrain=None) -> None:
        super(ResNet50, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        if not pretrain:
            self.vision_extractor = resnet50()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == 'IMAGENET1K':
            self.vision_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.vision_extractor.eval()
            self.fc = nn.Linear(1000, output_channel)
        elif pretrain == 'R3M':
            self.vision_extractor = load_r3m('res50')
            self.vision_extractor.eval()
            self.fc = nn.Linear(2048, output_channel).to('cuda')
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: B x 224 x 224 x 3
        # x = torch.permute(x, (0, 3, 1, 2))  # x: B x 3 x 224 x 224
        out = self.vision_extractor(x)
        out = self.fc(out)
        return out