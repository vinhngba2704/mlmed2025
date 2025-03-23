import torch
import numpy as np
from dataset import get_dataloader
from model import get_model

def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_score(outputs, masks)
            dice_scores.append(dice.item())

    print(f"Average Dice Score: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader("/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/images", "/media/Home/USTH/Year_3/MLinMedicine/prac2/training_set/annotations", batch_size=8)
    model = get_model()
    evaluate_model(model, dataloader, device= device)
