# Steel-Plates-Faults-Prediction-Using-Neural-Networks
This project focuses on predicting types of faults in steel plates using a neural network model. The dataset used contains various numeric attributes describing the characteristics of steel plates, and the goal is to classify these faults into one of seven categories. The exercise primarily utilizes numeric features, excluding the categorical ones.
Dataset Overview
The dataset contains the following variables:
Dependent Variables (Fault Types):
Pastry
Z_Scratch
K_Scratch
Stains
Dirtiness
Bumps
Other_Faults
Independent Variables (Numeric Attributes):
X_Minimum
X_Maximum
Y_Minimum
Y_Maximum
Pixels_Areas
X_Perimeter
Y_Perimeter
Sum_of_Luminosity
Minimum_of_Luminosity
Maximum_of_Luminosity
Length_of_Conveyer
Steel_Plate_Thickness 13-27. Various edge and luminosity indices
Note:
Among the independent variables, only the steel types (TypeOfSteel_A300 and TypeOfSteel_A400) are categorical, while all other attributes are numeric. In this project, the focus is on the numeric attributes, so the categorical variables were excluded.

# Problem Statement
The objective of this project is to build a neural network model that predicts the fault type in a steel plate using only numeric attributes. The performance of the model is evaluated using accuracy and a detailed classification report.

# Methodology
Data Preprocessing:
Dropped categorical variables (TypeOfSteel_A300, TypeOfSteel_A400) to focus on numeric attributes only.
One-hot encoded the target variable (7 fault types) for multi-class classification.
Split the data into training (80%) and testing (20%) sets.
Standardized the numeric features using StandardScaler.
Model Architecture:

# Input Layer: Features from the dataset.
Hidden Layers: Two hidden layers with 64 and 32 neurons respectively, using ReLU activation.
Output Layer: Softmax activation for multi-class classification of the 7 fault types.
# Training:
Used Adam optimizer and categorical cross-entropy as the loss function.
Trained the model for a maximum of 100 epochs, with early stopping applied to avoid overfitting.
Batch size: 32.
Validation data: 20% of the dataset (from the test split).
Evaluation:
Test Accuracy: 0.50.
Loss: 18.97.
A detailed classification report was generated, providing precision, recall, and F1-score for each fault type.
# Results
The model achieved an accuracy of 50% on the test data, with good performance in identifying the "Pastry" fault type (class 0).
However, the model struggled with other fault types, particularly "Z_Scratch" (class 1), where no correct predictions were made.
A detailed classification report is provided in the code to evaluate the performance across all fault types.
Classification Report Summary:
Class 0 (Pastry): Precision = 0.88, Recall = 0.54, F1-score = 0.67.
Class 1 (Z_Scratch): Precision, recall, and F1-score = 0.00, indicating the model failed to predict this fault.
Performance was poor across other fault types due to lack of sufficient representation in the dataset.
# Conclusion
The neural network model performed reasonably well for the "Pastry" fault type but struggled to predict the other fault types accurately. This could be due to an imbalanced dataset or insufficient features to differentiate between fault types.
# Further improvements could include:
Balancing the dataset.
Experimenting with different neural network architectures.
Feature engineering to enhance predictive power.
# Dependencies
Python 3.x
TensorFlow
Keras
scikit-learn
pandas
numpy
