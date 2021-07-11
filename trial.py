import sys
print(sys.executable)

import torch
import torch.nn as nn
import torchvision.models as models

model = getattr(models, "resnet101")(pretrained=True)
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)


y = model(torch.ones(1, 3, 224, 224).to("cpu"))
print(y.shape)
print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
