import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=2, padding=1),
                #nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=2, padding=1),
                #nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=2, padding=1),
                #nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(1,1))
                )
        
    def forward(self, x):
            x = self.conv_layer(x)
            return x.view(x.size(0), -1)
