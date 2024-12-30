import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import xml.etree.ElementTree as ET
from PIL import Image
import os
import time
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

class CustomCNN(nn.Module):
    def __init__(self, num_classes=40, dropout_prob=0.5, activation=nn.ReLU):
        super(CustomCNN, self).__init__()
        self.activation = activation()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Dropout
        self.dropout2d = nn.Dropout2d(p=dropout_prob)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Placeholder, corrected in forward
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for FC layers
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # Convolutional layers with BatchNorm, Activation, and MaxPooling
        x = self.activation(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.activation(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.activation(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.activation(self.bn4(self.conv4(x)))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Dynamically adjust fc1 based on the input size
        if not hasattr(self, "_fc1_initialized"):
            self.fc1 = nn.Linear(x.size(1), 512)
            self._fc1_initialized = True
        
        # Fully connected layers with dropout
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Function to retrieve a pre-defined model based on its type
def get_model_by_type(type):
    model_map = {
        'resnet': models.resnet18,
        'googlenet': models.googlenet,
        'efficientnet': models.efficientnet_b0,
        'densenet': models.densenet121,
        'customcnn': CustomCNN
    }
    return model_map.get(type)

# Function to choose an image file from a specified directory
def get_file(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    if len(files) == 0:  # Check if directory is empty
        print("Error: \"" + dir + "\" directory is empty")
        exit()

    print("Choose file: ")
    for i, f in enumerate(files, start=1):
        print(f"{i} - {f}")

    file_choose = int(input("File number: "))
    if file_choose > len(files):  # Validate user input
        print("Error: invalid file number")
        exit()

    return join(dir, files[file_choose - 1])

# Function to extract bounding box coordinates from an XML file
def get_xy_from_XML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return xmin, xmax, ymin, ymax

# Function to create training and testing datasets from image and annotation files
def create_sets(train_images, test_images, images_folder, xml_folder):
    train_folder = os.path.join(os.getcwd(), 'train')
    test_folder = os.path.join(os.getcwd(), 'test')

    # Create directories for training and testing sets
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    with open(train_images, 'r') as train_file, open(test_images, 'r') as test_file:
        for image in train_file:
            image = image.strip()
            src_path = join(images_folder, image)
            class_folder = os.path.splitext(os.path.basename(image))[0][:-4]
            os.makedirs(join(train_folder, class_folder), exist_ok=True)
            if os.path.exists(src_path):
                xmin, xmax, ymin, ymax = get_xy_from_XML(join(xml_folder, os.path.splitext(image)[0] + ".xml"))
                img = Image.open(src_path)
                cropped_image = img.crop((xmin, ymin, xmax, ymax))
                cropped_image.save(join(train_folder, class_folder, image))
        
        for image in test_file:
            image = image.strip()
            src_path = join(images_folder, image)
            class_folder = os.path.splitext(os.path.basename(image))[0][:-4]
            os.makedirs(join(test_folder, class_folder), exist_ok=True)
            if os.path.exists(src_path):
                xmin, xmax, ymin, ymax = get_xy_from_XML(join(xml_folder, os.path.splitext(image)[0] + ".xml"))
                img = Image.open(src_path)
                cropped_image = img.crop((xmin, ymin, xmax, ymax))
                cropped_image.save(join(test_folder, class_folder, image))

# Function to train a model using a given data loader
def train_model(train_loader, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        for i, data in tqdm(enumerate(train_loader, 0)):
            inputs, labels = data
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s')

# Function to test the model on test data
def test_model(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

# Function to save the model to a file
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Function to load a model from a file
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")
    return model

# Main function to start the program
def start():
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training and testing data
    train_data = datasets.ImageFolder(root='train', transform=transform)
    test_data = datasets.ImageFolder(root='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Choose to use an existing model or train a new one
    option = int(input("Choose option (1 - use existing model, 2 - train new model): "))
    if option == 1:
        model_filename = get_file("models")
        type = os.path.basename(model_filename).split('_')[0]
        model_class = get_model_by_type(type)
        if model_class:
            if type != 'customcnn': model = model_class(pretrained=True) 
            else: model = model_class()
            model = load_model(model, model_filename)
            train = input("Train this model more? [y/N]: ").strip().lower()
            if train == 'y':
                # Set up loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                num_epochs = int(input("Number of epochs: "))
                print(f"Starting training on model, epochs: {num_epochs}")
                train_model(train_loader, model, optimizer, criterion, num_epochs)
        else:
            print("Unknown model type: ", type)
            return
    elif option == 2:
        # Allow user to select a pre-defined model type
        print("Choose a model for testing:")
        print("1. ResNet-18")
        print("2. GoogLeNet")
        print("3. EfficientNet-B0")
        print("4. DenseNet-121")
        print("5. CustomCNN")
        model_choice = int(input("Your choice (1-5): "))

        if model_choice == 1:
            model = models.resnet18(pretrained=True)
        elif model_choice == 2:
            model = models.googlenet(pretrained=True)
        elif model_choice == 3:
            model = models.efficientnet_b0(pretrained=True)
        elif model_choice == 4:
            model = models.densenet121(pretrained=True)
        elif model_choice == 5:
            model = CustomCNN()
        else:
            print("Invalid choice.")
            return

        # Set up loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = int(input("Number of epochs: "))
        print(f"Starting training on model: {model_choice}, epochs: {num_epochs}")
        train_model(train_loader, model, optimizer, criterion, num_epochs)
    else:
        print("Invalid option number")

    # Optionally save the model
    save = input("Save model to a file? [y/N]: ").strip().lower()
    if save == 'y':
        model_filename = input("Enter a filename to save the trained model: ")
        save_model(model, os.path.join("models", f"{model.__class__.__name__.lower()}_{model_filename}.pth"))

    # Test the model and display accuracy
    test = input("Test model? [y/N]: ").strip().lower()
    if test == 'y':
        model.eval()  # Set model to evaluation mode
        correct, total = test_model(test_loader, model)
        print(f'Accuracy on test data: {100 * correct / total:.2f}%')

# Main entry point for the program
option = int(input("Choose option (1 - create training and testing image sets, 2 - test a model): "))
if option == 1:
    create_sets('ImageSplits/train.txt', 'ImageSplits/test.txt', 'JPEGImages', 'XMLAnnotations')
elif option == 2:
    start()
else:
    print("Invalid option number")