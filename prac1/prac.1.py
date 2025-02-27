import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from modules import plot_class_distribution, evaluate_model

# Load the data
df_train = pd.read_csv("/media/Personal/USTH/Year_3/MLinMedicine/prac1/mitbih_train.csv", header=None)
df_test = pd.read_csv("/media/Personal/USTH/Year_3/MLinMedicine/prac1/mitbih_test.csv", header=None)

# Divide into features and labels
X_train = df_train.drop(df_train.columns[-1], axis=1)
y_train = df_train[df_train.columns[-1]] 
X_test = df_test.drop(df_test.columns[-1], axis=1)
y_test = df_test[df_test.columns[-1]]

# Plot the class distribution of the dataset
plot_class_distribution(y_train, y_test)

# Label encoding
encoder = LabelEncoder()
encoder.fit_transform(y_train)
encoder.transform(y_test)

# Apply SMOTETomek to balance the training set
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_train_resample_tomek, y_train_resample_tomek = smote_tomek.fit_resample(X_train, y_train)

# Plot the class distribution of the resampled training set
plot_class_distribution(y_train_resample_tomek, y_test)

# Random Forest Classifier with SMOTETomek resampling
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resample_tomek, y_train_resample_tomek)
y_pred = rf.predict(X_test)

# Evaluate the model
evaluate_model(y_test, y_pred)