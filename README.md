# 🌸 Iris Flower Classification using ANN

An end-to-end machine learning project that predicts the species of an Iris flower using an Artificial Neural Network (ANN).

The project includes data preprocessing, feature scaling, label encoding, ANN model training, evaluation, and an interactive Streamlit application.

## 🚀 Live Demo

[Try the Iris Flower Classifier](iris-ann-classifier-fappvhzcuyux8appuwsrfpo.streamlit.app)

## 📌 Project Overview

The Iris dataset contains measurements of Iris flowers based on four features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The model predicts one of three species:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

## 🧠 Machine Learning Workflow

The project follows these steps:

1. Load and explore the Iris dataset
2. Check and preprocess the data
3. Encode the target labels
4. Split the data into training and testing sets
5. Standardize the input features
6. Build and train an Artificial Neural Network
7. Evaluate the trained model
8. Save the trained model and preprocessing objects
9. Build an interactive Streamlit application
10. Deploy the application

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Matplotlib
* Seaborn
* Joblib

## 📂 Project Structure

```text
iris-ann-classifier/
│
├── app.py
├── ann.keras
├── perceptron.pkl
├── scaler.pkl
├── encoder.pkl
├── iris.csv
├── iris_project.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Input Features

The model uses four numerical features:

| Feature      | Description                        |
| ------------ | ---------------------------------- |
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width  | Width of the sepal in centimeters  |
| Petal Length | Length of the petal in centimeters |
| Petal Width  | Width of the petal in centimeters  |

## 🤖 Model

The project contains a simple Perceptron implementation as well as an Artificial Neural Network.

The ANN uses:

* Input layer with 4 features
* Hidden layers for learning feature relationships
* Output layer with 3 classes
* Softmax activation for multiclass classification

The trained ANN is saved as:

```text
ann.keras
```

## ⚙️ Preprocessing

Before making predictions, the input features are standardized using `StandardScaler`.

The target species are encoded using `LabelEncoder`.

The trained preprocessing objects are saved for use during prediction:

```text
scaler.pkl
encoder.pkl
```

This ensures that the Streamlit application applies the same preprocessing used during model training.

## 🌐 Streamlit Application

The Streamlit application allows users to enter the four flower measurements and receive:

* Predicted Iris species
* Prediction confidence
* Probability of each class

## 📈 Future Improvements

* Add model performance visualizations
* Add confusion matrix visualization
* Improve the Streamlit UI
* Compare ANN performance with other classification algorithms
* Add more interactive visualizations

## 👨‍💻 Author

### Ghanshyam B
