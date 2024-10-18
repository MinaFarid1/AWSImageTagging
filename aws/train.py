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
import numpy as np
from sklearn.metrics import recall_score  # Import for recall calculation


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


# Define the Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.sigmoid(inputs) * targets + (1 - torch.sigmoid(inputs)) * (1 - targets)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# Define the model
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze the feature extraction layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),  # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),  # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # Output layer
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
    criterion = FocalLoss()  # Use Focal Loss instead of BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=args.lr)

    def evaluate(model, data_loader, criterion):
        model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)

        # Calculate recall
        recall = recall_score(all_labels, all_predictions, average='macro')
        return avg_loss, recall

    best_val_recall = 0.0  # Initialize the best validation recall

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
        val_loss, val_recall = evaluate(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{args.epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Recall: {val_recall:.4f}')  # Changed to show recall

        # Save the best model based on validation recall
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            CHECKPOINT_PATH = os.path.join(args.model_dir, 'best_model.pth')  # Define checkpoint path
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  Saving the best model with validation recall: {val_recall:.4f}")

    # Save final model
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