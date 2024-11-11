import torch
import torchvision.transforms as transforms
from PIL import Image

img_to_tensor = transforms.Compose([transforms.ToTensor()])

img = Image.open("hiragana_a_written.png")
img_ten = img_to_tensor(img)

print(img_ten.dtype)
print(img_ten.device)
print(img_ten.layout)
