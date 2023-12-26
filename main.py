import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from torchvision.models.resnet import ResNet152_Weights
from torchvision.transforms import functional as F

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = f'frame{self.img_labels.iloc[idx, 0]}.0.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = io.imread(img_path)

        # Convert image to 3 channels if it has 4 channels
        if image.shape[2] == 4:
            image = F.to_tensor(image)[:3, :, :]
        else:
            image = F.to_tensor(image)

        label = self.img_labels.iloc[idx, 1:5].values.astype('float32')
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, label

def train(num_epochs, dataloader, model, criterion, optimizer, device):
    print('Starting Training')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print the predicted values for the first image in the batch
            print(f"Batch {i + 1}, Image 1 Predicted Values: {outputs[0].detach().cpu().numpy()}")

            avg_loss = running_loss / (i + 1)  # Average loss for the epoch so far
            print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}')
            writer.add_scalar('training loss', avg_loss, epoch * len(dataloader) + i)
            running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
    else:
        print("MPS device not found.")
        device = torch.device("cpu")

    # Image Transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Custom Dataset
    dataset = CustomImageDataset(annotations_file='/Users/aiden/Desktop/Training1/data.csv',
                                 img_dir='/Users/aiden/Desktop/Training1',
                                 transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize ResNet-152 with the updated syntax
    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 4)  # Adjusting for 4 output values
    model = model.to(device)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Upgrade optimizer to ADAM

    # TensorBoard setup
    writer = SummaryWriter('runs/custom_dataset_experiment')

    # Train the model
    train(10, dataloader, model, criterion, optimizer, device)
