import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import smtplib
from email.mime.text import MIMEText
import cv2

# Configuração do dataset
class CuttingObjectsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        classes = {"cortante": 1, "nao_cortante": 0}
        
        for class_name in classes.keys():
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(classes[class_name])
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformações para normalização
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset e DataLoader
dataset = CuttingObjectsDataset(root_dir="dataset", transform=data_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Modelo pré-treinado
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binário: cortante/não cortante

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Função de Treinamento
def train_model(model, dataloader, epochs=4, lr=0.0001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Época {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "modelo_cortantes.pth")
    print("Modelo salvo!")

# Sistema de alerta por e-mail
def send_alert():
    print("Enviando alerta por e-mail...")

def predict(image):
    model.load_state_dict(torch.load("modelo_cortantes.pth"))
    model.eval()
    
    if isinstance(image, str):  # Se for caminho de arquivo, abre a imagem
        image = Image.open(image).convert("RGB")
    
    image = data_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    if prediction == 1:
        print("Objeto cortante detectado!")
        
        return True
    else:
        print("Seguro.")
        
        return False

# Se não existie o modelo, treina
if not os.path.exists("modelo_cortantes.pth"):
    treinamento = input("Deseja treinar o modelo? (s/n): ")
    if treinamento.lower() == "s":
        train_model(model, dataloader)