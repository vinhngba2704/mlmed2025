import torch
import numpy as np
from dataset import get_dataloader
from model import get_model
from circumference import calculate_circumference

def predict_image(model, image, device= torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image).squeeze(0).cpu().numpy()
    return output

def evaluate_circumference(model, dataset, num_samples=5):
    for i in range(num_samples):
        image, _ = dataset[i]
        predicted_mask = predict_image(model, image)
        circumference = calculate_circumference(predicted_mask)
        print(f"Sample {i+1}: Estimated Circumference = {circumference:.2f} pixels")

if __name__ == "__main__":
    dataset = get_dataloader("/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/images", "/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/annotations").dataset
    model = get_model()
    evaluate_circumference(model, dataset)
