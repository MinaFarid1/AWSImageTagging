import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import zipfile


# Custom Dataset class
class MultiLabelImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, categories_file, augment_classes=None, transform_aug=None, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.categories = pd.read_csv(categories_file)
        self.augment_classes = augment_classes
        self.transform_aug = transform_aug
        self.transform = transform
        self.num_classes = len(self.labels.columns) - 1

    def __len__(self):
        return len(self.labels) * (2 if self.augment_classes else 1)

    def __getitem__(self, idx):
        augment_idx = idx // 2 if self.augment_classes else idx
        img_name = self.labels.iloc[augment_idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        labels = torch.tensor(self.labels.iloc[augment_idx, 1:].values.astype(float), dtype=torch.float32)

        if self.augment_classes and any(labels[cls] == 1 for cls in self.augment_classes) and (idx % 2 == 1):
            if self.transform_aug:
                image = self.transform_aug(image)
        else:
            if self.transform:
                image = self.transform(image)

        return image, labels


# Define the model
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


# Define training and evaluation logic
def train(args):
    # Set up device and directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Unzip the training.zip file
    zip_file_path = os.path.join(args.data_dir, 'training.zip')
    unzip_dir = os.path.join(args.data_dir, 'train')
    
    if not os.path.exists(unzip_dir):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
    
    # Create dataset and dataloaders
    transform_regular = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = MultiLabelImageDataset(
        img_dir=unzip_dir,  # Path to unzipped images
        labels_file=os.path.join(args.data_dir, 'labels_train.csv'),  # Path to labels_train.csv
        categories_file=os.path.join(args.data_dir, 'categories.csv'),  # Path to categories.csv
        augment_classes=range(1, 80), 
        transform_aug=transform_aug,
        transform=transform_regular
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = full_dataset.num_classes

    # Model, Loss, and Optimizer
    model = MultiLabelClassifier(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=args.lr)

    def evaluate(model, data_loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0) * num_classes
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0) * num_classes
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{args.epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))


# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)

    # SageMaker directories
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()
    train(args)