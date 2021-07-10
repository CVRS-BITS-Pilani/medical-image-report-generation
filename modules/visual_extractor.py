import torch
import torch.nn as nn
import torchvision.models as models
from modules.hrnet.hrnet import HRNet

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        if self.visual_extractor == "hrnet":
            model = HRNet(32, 17, 0.1)
            model.load_state_dict(
                # torch.load('./weights/pose_hrnet_w48_384x288.pth')
                torch.load('./weights/pose_hrnet_w32_256x192.pth')
            )
            print("HR-Net loaded!")

            modules = []
            modules.append(model)
            self.convHR = nn.Conv2d(17, 2048, 32, 4)
            modules.append(self.convHR)
            
            self.model = nn.Sequential(*modules)

        else:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)

        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
