import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloader
from model import get_model

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "unet_trained.pth")
    print("Model saved as unet_trained.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader("/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/images", "/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/annotations", batch_size=8)
    model = get_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10, device= device)
