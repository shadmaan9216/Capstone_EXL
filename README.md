
# Credit Card Fraud Detection Web App

This repository contains a Streamlit web application for detecting credit card fraud. The app uses various machine learning models to classify whether a transaction is fraudulent based on anonymized data.

## Table of Contents
1. [About](#about)
2. [Features](#features)
3. [Algorithms Used](#algorithms-used)
4. [Data](#data)
5. [How to Run the App](#how-to-run-the-app)
6. [How to Run the App Using Docker](#how-to-run-the-app-using-docker)
7. [Usage](#usage)

## 1. About
The Credit Card Fraud Detection Web App utilizes machine learning algorithms to identify potentially fraudulent credit card transactions. This project aims to provide a user-friendly interface where users can input transaction data and obtain predictions on whether the transaction is legitimate or fraudulent.

The dataset used for training the models consists of anonymized credit card transactions, where features are labeled as V1, V2, ..., V28 due to the sensitive nature of the data. The project implements Principal Component Analysis (PCA) for dimensionality reduction and uses oversampling techniques like SMOTE to handle class imbalance.

## 2. Features
- **Prediction Interface**: Input transaction data to predict if a transaction is fraudulent.
- **Multiple Algorithms**: Three models for undersampled data and three for oversampled data (Logistic Regression, Decision Tree, Random Forest).
- **Model Download**: Ability to download the trained model (if it is not already available locally).
- **Data Visualizations**: Insightful visualizations related to credit card fraud patterns.
- **Interactive UI**: Powered by Streamlit with enhanced HTML and CSS styling.
- **Scalable**: PCA-applied features and scaling integrated for accurate prediction.

## 3. Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest

Data preprocessing involves:
- SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
- Scaling for normalized input before applying machine learning models.
- PCA to handle high-dimensional data (V1 to V28 features).

## 4. Data
The dataset used in this project comes from anonymized credit card transactions. The columns V1 to V28 represent features extracted using Principal Component Analysis (PCA). The dataset includes the Time, Amount, and Class columns, where Class indicates whether a transaction is fraudulent (1) or legitimate (0).

For more information on the dataset, you can refer to the original dataset source.

## 5. How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/shadmaan9216/Capstone_EXL.git
   ```
2. Navigate to the project directory:
   ```bash
   cd exl_project_team_2/final-app
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 6. How to Run the App Using Docker
**Important** - Make sure you have installed Docker on your PC:
- Linux: [Docker](https://docs.docker.com/engine/install/)
- Windows/Mac: [Docker Desktop](https://docs.docker.com/desktop/)

1. Start Docker:
   - **Linux (Home Directory)**:
     ```bash
     sudo systemctl start docker
     ```
   - **Windows**: Start Docker engine from Docker Desktop.
2. Log in to Docker Hub from the terminal. You can log in with your password or access token:
   ```bash
   docker login
   ```
3. Pull the docker image from Docker Hub:
   ```bash
   docker pull shramanjain98/webapp:webapp
   ```
4. Run the Docker Container:
   ```bash
   docker run -p 8501:8501 shramanjain98/webapp:webapp
   ```

## 7. Usage
- Launch the web app locally.
- Input the transaction data including V1, V2, ..., V28, Amount, and Time.
- Click **Predict** to check if the transaction is fraudulent or not.
- Explore visualizations and the explanation of models used.
