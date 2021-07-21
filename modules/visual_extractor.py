import torch
import torch.nn as nn
import torchvision.models as models
from modules.hrnet.hrnet import HRNet
from modules.hrnet.cbam import CBAM

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
            
            self.cbam1 = CBAM(64, 16)
            self.cbam2 = CBAM(128, 16)
            self.cbam3 = CBAM(256, 16)

            self.conv1= nn.Conv2d(32, 512, 8, 8)      
            self.conv2= nn.Conv2d(64, 512, 4, 4)  
            self.conv3= nn.Conv2d(128, 512, 2, 2) 
            self.conv4= nn.Conv2d(256, 512, 1, 1)   

            self.model = model

        else:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)

        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats, first, second, third = self.model(images)

        if self.visual_extractor == "hrnet":
            patch_feats = self.conv1(patch_feats)

            # first = self.conv2(self.cbam1(first))
            # second = self.conv3(self.cbam2(second))
            # third = self.conv4(self.cbam3(third))

            # patch_feats = torch.cat((patch_feats, first, second, third), 1)

        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
