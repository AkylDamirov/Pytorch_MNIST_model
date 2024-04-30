import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from model import Classifier

import matplotlib.pyplot as plt

#declare hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 5

#data loading and transforms
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))
])

train_dataset = MNIST(root='.data/', train=True, download=True, transform=data_transforms)
test_dataset = MNIST(root='.data/', train=False, download=True, transform=data_transforms)

#dividing our data into batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Classifier(input_size=784, num_classes=10)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)

#training loop

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    total_epoch_loss = 0
    steps = 0
    for batch in iter(train_loader):
        images, labels = batch
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        #calls the model.forward() for generating predictions
        predictions = model(images)
        #calculate cross entry loss
        loss = criterion(predictions, labels)
        #clears gradient values from previous batch
        optimizer.zero_grad()
        #computes backprop gradient based on the loss
        loss.backward()
        #optimizes the model weights
        optimizer.step()

        steps +=1
        total_epoch_loss += loss.item()

    print(f'Epoch {epoch+1} / {NUM_EPOCHS}: AVERAGE LOSS: {total_epoch_loss/steps}')

#Evaluting our model perfomance
correct_predictions = 0
total_predictions = 0
for batch in iter(test_loader):
    images, labels = batch
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    predictions = model(images)
    #Taking predicted label with highest probability
    predictions = torch.argmax(predictions, dim=1)
    correct_predictions += (predictions == labels).sum().item()
    total_predictions += labels.shape[0]

print(f"\nTEST ACCURACY: {((correct_predictions / total_predictions) * 100):.2f}")















