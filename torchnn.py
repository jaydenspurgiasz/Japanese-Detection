# Import
import torch
import onnx
import os
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision import transforms

import cv2
import numpy

from customDatasets import HiraganaDataset

# Import data

# train_data = datasets.FashionMNIST(root="data", download=True, train=True, transform=ToTensor())
# test_data = datasets.FashionMNIST(root="data", train=False, download=True)
# train_data = datasets.MNIST(root="./data/HiraganaMNIST/", train=True, download=True, transform=ToTensor())
# test_data = datasets.MNIST(root="./data/HiraganaMNIST/", train=True, download=True)

data = HiraganaDataset("./data/HiraganaDataset/train")
train_data, test_data, useless_data = random_split(data, [10000, 3800, 0])

# test_data = HiraganaDataset("./data/HiraganaMNIST/MNIST/test")

dataset = DataLoader(train_data, 20)  # train in sets of 32
# 1, 28, 28 - classes 0-9

# Image CLassifier Neural Network


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(  # Neural network layers, steps?
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*(128-8)*(127-8), 69)  # shaved pixels, 10 outputs
        )

    def forward(self, x):
        return self.model(x)


# Create Instance of neural network, loss, optimizer
clf = ImageClassifier().to("cuda")
# Optimizer: pass classifier, learning rate
opt = Adam(clf.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


def test_loss():
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f, weights_only=True))
    
    img_array = ["hiragana_a.png", "hiragana_a_1.png", "hiragana_o.png", "hiragana_zu.png"]
    ans_array = [0, 0, 39, 68]
    prediciton_array = []
    correct = 0

    for i in range(len(img_array)):
        prediction = torch.argmax(clf(ToTensor()(Image.open(img_array[i])).unsqueeze(0).to("cuda")))
        prediciton_array.append(prediction)
        if prediction == ans_array[i]:
             correct += 1
    print(f"Correct: {correct}/4  |  Predictions: {prediciton_array}")
    

def train():
        # Training, comment when running, uncomment when training
        for epoch in range(30):  # Train for 10 epochs
            for batch in dataset:
                X, y = batch
                X, y = X.to("cuda"), y.to("cuda")
                yhat = clf(X)
                loss = loss_fn(yhat, y)
                del X, y, yhat

                # Apply backdrop
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            # Save trained model into a file
            with open("model_state.pt", "wb") as f:
                save(clf.state_dict(), f)

            print(f"Epoch: {epoch} loss is {loss.item()}")
            print(test_loss())

        # # Save trained model into a file
        # with open("model_state.pt", "wb") as f:
        #     save(clf.state_dict(), f)

def test():

    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    correct = 0
    incorrect = 0

    tensorToImage = transforms.ToPILImage()

    for test_image in test_data:
        img = tensorToImage(test_image[0])
        img_tensor = ToTensor()(img).unsqueeze(0).to("cuda") # Unsqeeze image prediction

        prediction = torch.argmax(clf(img_tensor))
        if prediction == test_image[1]:
            correct+=1
        else:
            incorrect+=1
        
        print(f"Prediction is: {prediction} and the real value is: {test_image[1]}")

    print(f"There were {correct} correct predictions and {incorrect} incorrect predictions.")
    print(f"The percentage correct was: {(correct/(correct+incorrect)*100):.2f}%")

def run():

        # Load and run model

        with open("model_state.pt", "rb") as f:
            clf.load_state_dict(load(f, weights_only=True))

        img = Image.open("hiragana_a.png")
        img_tensor = ToTensor()(img).unsqueeze(0).to("cuda")  # Unsqeeze image prediction

        print(torch.argmax(clf(img_tensor)))


#Function to Convert to ONNX 
def Convert_ONNX(): 
    clf = ImageClassifier()
    with open("model_state.pt", "rb") as f:
            clf.load_state_dict(load(f, map_location=torch.device("cpu")))
    torch.onnx.export(clf, torch.randn((1, 127, 128), dytpe=torch.float32, layout=torch.strided), "HiraganaModel1.onnx", export_params=True)
    model = onnx.load("HiraganaModel1.onnx")
    onnx.checker.check_model(model)


# Main
if __name__ == "__main__":
    train()
