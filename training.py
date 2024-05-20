import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Compose, Lambda
import matplotlib.pyplot as plt
from vit import Vit
from tqdm import tqdm
import numpy as np


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform= ToTensor())

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform= ToTensor())
#Vit parameters
in_channels = 1

hidden_size = 16

img_size = (1,28,28)

num_classes = 10

patch_size = 4
batch_size = 32
epochs=5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Vit(in_channels, hidden_size, img_size, num_classes, patch_size)

#Dataloader
dataloader_training = DataLoader(training_data,batch_size=batch_size,shuffle=True)

dataloader_testing = DataLoader(test_data,batch_size=1,shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cross_entropy = torch.nn.CrossEntropyLoss()


epoch_loss = []
epoch_accuracy_training = []


for name,param in model.named_parameters():
    print(f"{name} -> {param.shape}")

for epoch in (range(epochs)):

    training_loss = []
    train_acc_sum = 0
    counter = 0
    for x,y in tqdm(dataloader_training):
        output = model(x)

        loss = cross_entropy(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output, 1)
        train_accuracy = torch.sum(predicted == y)

        counter += x.shape[0]
        train_acc_sum += train_accuracy
        training_loss.append(loss.detach().cpu().item())
    
    print(f"Total element trained {counter}")
    print(f"Accuracy training {train_acc_sum}")
    print(f"Avg accuracy training {train_acc_sum/counter}")
    print(f"Epoch {epoch} -> {np.mean(training_loss)}")
    epoch_loss.append(np.mean(training_loss))
    epoch_accuracy_training.append(train_accuracy/counter)



testing_loss =[]
testing_acc = 0
with torch.no_grad():
    for x, y in tqdm(dataloader_testing):
        output = model(x)
        #loss = cross_entropy(output,y)
        #testing_loss.append(loss.item())
        predicted_test = torch.argmax(output, 1)
        if (predicted_test == y):
            testing_acc+=1
    

print(f"Len testing {len(dataloader_testing)}")
print(f"Avg accuracy testing {testing_acc/len(dataloader_testing)}")




    