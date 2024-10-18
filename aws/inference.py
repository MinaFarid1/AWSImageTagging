import os
import boto3
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import logging
import io
import json
# Logging for debugging and catching errors
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("inference.log")  
    ]
)

logger = logging.getLogger(__name__)

# Function to load categories from S3
def load_categories_from_s3(bucket_name, object_key):
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        categories_df = pd.read_csv(obj['Body'], header=None)
        return categories_df
    except Exception as e:
        logger.error(f"Failed to load categories from S3: {str(e)}")
        raise

# Define S3 bucket and object key
bucket_name = 'bucket_name'  
object_key = 'models/categories.csv'

category_names = None

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Load model
def model_fn(model_dir):
    try:
        # Load the model for inference
        model = MultiLabelClassifier(num_classes=80)
        model_path = os.path.join(model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Load categories once and store in memory
        global category_names
        if category_names is None:
            categories_df = load_categories_from_s3(bucket_name, object_key)
            category_names = categories_df.iloc[:, 0].tolist()

        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
# Preprocess the input
def input_fn(request_body, content_type):
    try:
        if content_type == 'application/x-image':
            # Read the byte stream directly
            img = Image.open(io.BytesIO(request_body)).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = transform(img)
            return img.unsqueeze(0)
        else:
            raise ValueError(f'Unsupported content type: {content_type}')
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise


def predict_fn(input_data, model):
    try:
        with torch.no_grad():
            outputs = model(input_data)
            probabilities = torch.sigmoid(outputs).squeeze().numpy()
        return probabilities
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)} | Input data: {input_data}")
        raise


def output_fn(prediction, content_type='application/json'):
    try:
        categories = category_names

        # Get top 5 predictions
        top5_indices = prediction.argsort()[-5:][::-1]
        top5_probabilities = prediction[top5_indices]
        top5_categories = [categories[i] for i in top5_indices]

        # Create a dictionary where keys are category names and values are probabilities
        top5_predictions = {top5_categories[i]: float(top5_probabilities[i]) for i in range(5)}

        # Wrap the top 5 predictions inside the "probabilities" key
        result = {
            "probabilities": top5_predictions
        }

        # Ensure the result is returned as a JSON string
        return json.dumps(result)  # Return JSON-encoded string
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise