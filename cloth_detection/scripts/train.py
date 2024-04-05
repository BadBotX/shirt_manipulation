import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import SGD
import utils  
from pathlib import Path
import os
from torchvision.transforms import ToTensor
import json
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.ann_files = [file for file in self.ann_dir.glob('*.json')]

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        ann_path = self.ann_files[idx]
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = self.img_dir / ann['imagePath']
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        boxes = []
        for shape in ann['shapes']:
            if shape['shape_type'] == 'rectangle':
                xmin, ymin = shape['points'][0]
                xmax, ymax = shape['points'][1]
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target

def get_transform(train):
    transforms = []
    # Add more transforms as needed
    if train:
        # Add any transformations for training here
        pass
    
    # Convert PIL image to Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    # Load the training and validation datasets
    current_path = Path(__file__).parent
    dataset = CustomDataset(current_path.parent / 'data'/'raw', current_path.parent / 'data'/'annotations', get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)

    # Load a model pre-trained on COCO and modify it to match your dataset's number of classes
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (your object) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the right device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss: {losses.item()}")

    trained_model_dir = current_path.parent / 'trained_models'
    trained_model_filename = 'model_weights.pth'
    model_path = os.path.join(trained_model_dir, trained_model_filename)
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
