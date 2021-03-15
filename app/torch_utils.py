import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

#transfer learning
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 128)

#load my trained weights
device = torch.device('cpu')
model.load_state_dict(torch.load('weights.pth', map_location=device))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 3, 224, 224)
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted