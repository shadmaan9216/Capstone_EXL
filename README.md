# Credit Card Fraud Detection Web App

This repository contains a Streamlit web application for detecting credit card fraud. The app uses various machine learning models to classify whether a transaction is fraudulent based on anonymized data.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Algorithms Used](#algorithms-used)
- [Data](#data)
- [How to Run the App](#how-to-run-the-app)
- [How to Run the App Using Docker](#how-to-run-the-app-using-docker)
- [Usage](#usage)

## About
The Credit Card Fraud Detection Web App utilizes machine learning algorithms to identify potentially fraudulent credit card transactions. This project aims to provide a user-friendly interface where users can input transaction data and obtain predictions on whether the transaction is legitimate or fraudulent.

The dataset used for training the models consists of anonymized credit card transactions, where features are labeled as V1, V2, ..., V28 due to the sensitive nature of the data. The project implements Principal Component Analysis (PCA) for dimensionality reduction and uses oversampling techniques like SMOTE to handle class imbalance.

## Features
- **Prediction Interface**: Input transaction data to predict if a transaction is fraudulent.
- **Multiple Algorithms**: Three models for undersampled data and three for oversampled data (Logistic Regression, Decision Tree, Random Forest).
- **Model Download**: Ability to download the trained model (if it is not already available locally).
- **Data Visualizations**: Insightful visualizations related to credit card fraud patterns.
- **Interactive UI**: Powered by Streamlit with enhanced HTML and CSS styling.
- **Scalable**: PCA-applied features and scaling integrated for accurate prediction.

## Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest

Data preprocessing involves:
- SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
- Scaling for normalized input before applying machine learning models.
- PCA to handle high-dimensional data (V1 to V28 features).

## Data
The dataset used in this project comes from anonymized credit card transactions. The columns V1 to V28 represent features extracted using Principal Component Analysis (PCA). The dataset includes the Time, Amount, and Class columns, where Class indicates whether a transaction is fraudulent (1) or legitimate (0).

For more information on the dataset, you can refer to the original dataset source.

## How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/Shraman-jain/exl_project_team_2.git
