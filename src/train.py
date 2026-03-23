import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load data
iris = load_iris()
X = iris.data # shape (150, 4)
Y = iris.target # shape (150,)
print (iris.feature_names, iris.target_names)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

print("predicitons", Y_pred[:5])
print("True labels", Y_test[:5])

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("accuracy:", accuracy)

cm = confusion_matrix(Y_test, Y_pred) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

# Create outputs directory in iris-classifier folder if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))  # iris-classifier/src
project_dir = os.path.dirname(script_dir)  # iris-classifier
outputs_dir = os.path.join(project_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

output_path = os.path.join(outputs_dir, 'confusion_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to {output_path}")









