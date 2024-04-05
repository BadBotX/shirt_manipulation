import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_pil_image, to_tensor
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def detect_regions_of_interest(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Assuming detection threshold of 0.5, adjust as necessary
    pred_boxes = [box for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']) if score > 0.5]
    return image, pred_boxes

def detect_shoulder_points_within_roi(image, boxes):
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    shoulder_points = []
    
    #place holder

    return shoulder_points

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_path = Path(__file__).parent
    model_path = current_path.parent / 'trained_models'/ 'model_weights.pth'
    image_path = current_path.parent / 'data'/ 'test'
    
    # Load the pre-trained model and detect regions of interest
    model = load_model(model_path, device)
    image, pred_boxes = detect_regions_of_interest(model, image_path, device)
    
    # Detect shoulder points within the detected regions
    shoulder_points = detect_shoulder_points_within_roi(image, pred_boxes)
    
    # For visualization
    for point in shoulder_points:
        cv2.circle(np.array(image), point, 5, (0, 255, 0), -1)
    cv2.imshow("Shoulder Points", np.array(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()