import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Function to plot the class distribution of the dataset
def plot_class_distribution(y_train, y_test):
    plt.figure(figsize=(12, 6))

    # First subplot for training set
    plt.subplot(1, 2, 1)
    bars_train = plt.bar(y_train.unique(), 
                         y_train.value_counts(), 
                         color='b')
    plt.xlabel("Labels")
    plt.ylabel("Number of values")
    plt.title("Training Set Distribution")
    for bar in bars_train:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            round(bar.get_height(), 2),
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Second subplot for testing set
    plt.subplot(1, 2, 2)
    bars_test = plt.bar(y_test.unique(), 
                        y_test.value_counts(), 
                        color='orange')
    plt.xlabel("Labels")
    plt.ylabel("Number of values")
    plt.title("Testing Set Distribution")
    for bar in bars_test:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            round(bar.get_height(), 2),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()  # Adjust layout to avoid overlapping elements
    plt.show(block = True)

def evaluate_model(y_true, y_pred, class_names=None):
    # Print metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    
    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()