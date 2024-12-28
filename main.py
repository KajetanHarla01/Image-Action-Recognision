import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import xml.etree.ElementTree as ET
from PIL import Image
import os
import time

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 40)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def get_xy_from_XML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return xmin, xmax, ymin, ymax

def create_sets(train_images, test_images, images_folder, xml_folder):
    train_folder = os.path.join(os.getcwd(), 'train')
    test_folder = os.path.join(os.getcwd(), 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    with open(train_images, 'r') as train_file, open(test_images, 'r') as test_file:
        for image in train_file:
            image = image.strip()
            src_path = os.path.join(images_folder, image)
            class_folder = os.path.splitext(os.path.basename(image))[0][0:-4] 
            os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
            if os.path.exists(src_path) and not os.path.exists(os.path.join(train_folder, class_folder, image)):
                xmin, xmax, ymin, ymax = get_xy_from_XML(os.path.join(xml_folder, os.path.splitext(os.path.basename(image))[0] + ".xml"))
                img = Image.open(src_path)
                cropped_image = img.crop((xmin, ymin, xmax, ymax))
                cropped_image.save(os.path.join(train_folder, class_folder, image))
        
        for image in test_file:
            image = image.strip()
            src_path = os.path.join(images_folder, image)
            class_folder = os.path.splitext(os.path.basename(image))[0][0:-4]
            os.makedirs(os.path.join(test_folder, class_folder), exist_ok=True)
            if os.path.exists(src_path) and not os.path.exists(os.path.join(test_folder, class_folder, image)):
                xmin, xmax, ymin, ymax = get_xy_from_XML(os.path.join(xml_folder, os.path.splitext(os.path.basename(image))[0] + ".xml"))
                img = Image.open(src_path)
                cropped_image = img.crop((xmin, ymin, xmax, ymax))
                cropped_image.save(os.path.join(test_folder, class_folder, image))

def train_model(train_loader, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in train_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s')

def test_model(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

def customized_CCN_network():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(root='train', transform=transform)
    test_data = datasets.ImageFolder(root='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    model = models.resnet18(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, optimizer, criterion, num_epochs=15)
    correct, total = test_model(test_loader, model)

    print(f'Accuracy on test data: {100 * correct / total:.2f}%')

option = int(input("Choose option (1 - create training and testing image sets, 2 - customized CNN network, 3 - pre-trained deep learning network): "))
if option == 1:
    create_sets('ImageSplits/train.txt', 'ImageSplits/test.txt', 'JPEGImages', 'XMLAnnotations')
elif option == 2:
    customized_CCN_network()
elif option == 3:
    pass
else:
    print("Invalid option number")
