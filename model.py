from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn

class MyResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
    #замена слоя
    self.model.fc = nn.Linear(512, 100)
    #разморозка
    for i in self.model.parameters():
      i.requires_grad = False

    self.model.fc.weight.requires_grad = True
    self.model.fc.bias.requires_grad = True

  def forward(self, x):
    return self.model(x)