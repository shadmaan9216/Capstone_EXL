Credit Card Fraud Detection Web App
This repository contains a Streamlit web application for detecting credit card fraud. The app uses various machine learning models to classify whether a transaction is fraudulent based on anonymized data.

Table of Contents
About
Features
Algorithms Used
Data
How to Run the App
How to Run the App Using Docker
Usage
About
The Credit Card Fraud Detection Web App utilizes machine learning algorithms to identify potentially fraudulent credit card transactions. This project aims to provide a user-friendly interface where users can input transaction data and obtain predictions on whether the transaction is legitimate or fraudulent.

The dataset used for training the models consists of anonymized credit card transactions, where features are labeled as V1, V2, ..., V28 due to the sensitive nature of the data. The project implements Principal Component Analysis (PCA) for dimensionality reduction and uses oversampling techniques like SMOTE to handle class imbalance.

Features
Prediction Interface: Input transaction data to predict if a transaction is fraudulent.
Multiple Algorithms: Three models for undersampled data and three for oversampled data (Logistic Regression, Decision Tree, Random Forest).
Model Download: Ability to download the trained model (if it is not already available locally).
Data Visualizations: Insightful visualizations related to credit card fraud patterns.
Interactive UI: Powered by Streamlit with enhanced HTML and CSS styling.
Scalable: PCA-applied features and scaling integrated for accurate prediction.
Algorithms Used
Logistic Regression
Decision Tree
Random Forest
Data preprocessing involves:

SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
Scaling for normalized input before applying machine learning models.
PCA to handle high-dimensional data (V1 to V28 features).
Data
The dataset used in this project comes from anonymized credit card transactions. The columns V1 to V28 represent features extracted using Principal Component Analysis (PCA). The dataset includes the Time, Amount, and Class columns, where Class indicates whether a transaction is fraudulent (1) or legitimate (0).

For more information on the dataset, you can refer to the original dataset source.

How to Run the App
Clone this repository:
git clone https://github.com/Shraman-jain/exl_project_team_2.git
Navigate to the project directory:
cd exl_project_team_2/final-app
Install the required dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
How to Run the App Using Docker
Important - Make sure you have installed Docker on your PC:

Linux: Docker
Windows/Mac: Docker Desktop
Start Docker:

Linux (Home Directory):
sudo systemctl start docker
Windows: You can start Docker engine from Docker Desktop.
Log in to Docker Hub from the terminal. You can log in with your password or access token:

docker login
Pull the docker image from the docker hub:

docker pull shramanjain98/webapp:webapp
Run the Docker Container:

docker run -p 8501:8501 shramanjain98/webapp:webapp
Usage
Launch the web app locally.
Input the transaction data including V1, V2, ..., V28, Amount, and Time.
Click Predict to check if the transaction is fraudulent or not.
Explore visualizations and the explanation of models used.
