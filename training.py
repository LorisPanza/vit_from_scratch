import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from vit import Vit
from tqdm import tqdm
import numpy as np


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def main():
    # -----------------------------------------------------------
    # Load FashionMNIST dataset for training and testing
    # -----------------------------------------------------------
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # -----------------------------------------------------------
    # Model and training parameters
    # -----------------------------------------------------------
    in_channels = 1                  # Grayscale images
    hidden_size = 16                 # Embedding dimension
    img_size = (1, 28, 28)           # Image dimensions (C, H, W)
    num_classes = 10                 # Number of FashionMNIST classes
    patch_size = 4                   # Patch size for ViT
    batch_size = 32                  # Training batch size
    epochs = 5                       # Number of training epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------
    # Initialize the Vision Transformer model and move to device
    # -----------------------------------------------------------
    model = Vit(in_channels, hidden_size, img_size, num_classes, patch_size).to(device)

    # -----------------------------------------------------------
    # Create DataLoaders for training and testing
    # -----------------------------------------------------------
    dataloader_training = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    dataloader_testing = DataLoader(test_data, batch_size=1, shuffle=False)

    # -----------------------------------------------------------
    # Set optimizer and loss function
    # -----------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------------------------------------
    # Lists for tracking training metrics
    # -----------------------------------------------------------
    epoch_loss = []
    epoch_accuracy_training = []

    # -----------------------------------------------------------
    # Print model parameter names and their shapes
    # -----------------------------------------------------------
    for name, param in model.named_parameters():
        print(f"{name} -> {param.shape}")

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        training_loss = []
        correct_predictions = 0
        total_samples = 0

        for x, y in tqdm(dataloader_training, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x)

            # Compute loss
            loss = criterion(output, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == y).sum().item()
            total_samples += y.size(0)

            training_loss.append(loss.item())

        avg_loss = np.mean(training_loss)
        avg_accuracy = correct_predictions / total_samples

        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Training Accuracy = {avg_accuracy:.4f}")

        epoch_loss.append(avg_loss)
        epoch_accuracy_training.append(avg_accuracy)

    # -----------------------------------------------------------
    # Evaluation on the test set
    # -----------------------------------------------------------
    model.eval()
    testing_correct = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader_testing, desc="Testing"):
            x, y = x.to(device), y.to(device)

            output = model(x)
            predicted = torch.argmax(output, dim=1)

            if predicted.item() == y.item():
                testing_correct += 1

    total_test_samples = len(dataloader_testing)
    test_accuracy = testing_correct / total_test_samples

    print(f"Test Accuracy: {test_accuracy:.4f} ({testing_correct}/{total_test_samples})")


if __name__ == '__main__':
    main()
    