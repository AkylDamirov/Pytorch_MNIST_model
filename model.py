import torch
import torch.nn as nn

#defining our classification model
class Classifier(nn.Module):
    def __init__(self, input_size,num_classes):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        output = self.output_layer(x)
        return output



