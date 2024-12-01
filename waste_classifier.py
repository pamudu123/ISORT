import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class WasteClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(WasteClassifier, self).__init__()
        # Load pretrained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)