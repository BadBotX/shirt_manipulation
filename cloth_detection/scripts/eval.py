import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os

def load_model(num_classes, model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def transform_image(image):
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image)

def display_predictions(image, predictions, delay=5000):
    for box in predictions['boxes']:
        x1, y1, x2, y2 = box.int().cpu().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Predictions', image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def evaluate(model, dataset_dir, device):
    for image_file in sorted(os.listdir(dataset_dir)):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(dataset_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform_image(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(image_tensor)[0]
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            display_predictions(image_cv, prediction)

def main():
    current_path = Path(__file__).parent
    model_path = current_path.parent / 'trained_models'/ 'model_weights.pth'
    dataset_dir = current_path.parent / 'data'/ 'test'
    num_classes = 2  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(num_classes, model_path, device)
    evaluate(model, dataset_dir, device)
if __name__ == "__main__":
    main()