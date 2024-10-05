import pandas as pd
import pytest
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE



@pytest.fixture
def load_data():
    # Load a small portion of data for testing
    return pd.read_csv('creditcard.csv')

@pytest.fixture
def origional_data(load_data):
    df = load_data

    X = df.drop(['Class'], axis=1)
    Y = df["Class"]
    
    X_data=X.values
    Y_data=Y.values
    return train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

#Under Sampled data
@pytest.fixture
def under_sampling_data(load_data):
    df = load_data
    # Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
    # RobustScaler is less prone to outliers.

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    # Amount and Time are Scaled!

    # Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.
    # Lets shuffle the data before creating the subsamples

    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    legit_df = df.loc[df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, legit_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    X = new_df.drop(columns = 'Class')
    y = new_df['Class']

    return train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)

#Over Sampled data
@pytest.fixture
def over_sampling_data(load_data):
    df = load_data

    X = df.drop('Class',axis=1)
    y = df['Class']
    X_res,y_res = SMOTE().fit_resample(X,y)
    y_res.value_counts()
    return train_test_split(X_res,y_res,test_size=0.20,random_state=42)




# test_data_preprocessing
# Test case for loading the dataset
def test_data_loading(load_data):
    data = load_data
    
    # Assert that data is loaded and has expected number of columns
    assert not data.empty, "Dataset is empty"

# Check if the required columns exist
def test_columns(load_data):
    data = load_data
    # Check if the required columns exist
    required_columns = ['Time', 'Amount', 'Class']
    for col in required_columns:
        assert col in data.columns, f"{col} is missing in the dataset"

# Ensure the Class column has 0 (non-fraud) and 1 (fraud)
def test_fraud_class_balance(load_data):
    data = load_data
    # Ensure the Class column has 0 (non-fraud) and 1 (fraud)
    assert set(data['Class'].unique()) == {0, 1}, "Class column is unbalanced"


#Test case - Trying random forest for original imbalanced data
def test_randomForest_original_imbalanced_data(load_data):
    data = load_data

    # Seperate X and Y
    X = data.drop(['Class'], axis=1)
    Y = data['Class']

    #test case to verify if the features (X) and target (Y) are separated correctly.
    # Assert that X and Y have the correct number of rows and columns
    assert X.shape[0] == Y.shape[0], "Mismatch in rows between X and Y"
    assert 'Class' not in X.columns, "Target column 'Class' should not be in X"

    data = load_data
    X = data.drop(['Class'], axis=1).values
    Y = data['Class'].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Expected sizes with a tolerance of ±1 row
    expected_train_size = math.ceil(0.8 * len(X))
    expected_test_size = math.ceil(0.2 * len(X))

    #test case - ensure that the train-test split works as expected.
    # Assert the split is done correctly with tolerance
    assert abs(X_train.shape[0] - expected_train_size) <= 1, "Train-test split size is incorrect"
    assert abs(X_test.shape[0] - expected_test_size) <= 1, "Test set size is incorrect"

    # Train Random Forest
    rfc = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt', random_state=42)
    rfc.fit(X_train, Y_train)
    
    # Assert the model has been trained
    assert rfc is not None, "Model training failed"
    
    y_pred = rfc.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(Y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low"
    assert prec > 0.7, "Precision is too low"
    assert rec > 0.7, "Recall is too low"
    assert f1 > 0.7, "F1-Score is too low"


###---------------------------------------------------------------------------
###----------------Under Sampling---------------------------------------------
###---------------------------------------------------------------------------
#Logistic Regression
def test_logisticRegression_underSampling(under_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = under_sampling_data


    #Testing model on under sampled data
    # Train Logistc Regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_logisticRegression_underSampling)"
    assert prec > 0.8, "Precision is too low (test_logisticRegression_underSampling)"
    assert rec > 0.8, "Recall is too low (test_logisticRegression_underSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_logisticRegression_underSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_logisticRegression_underSampling original data)"
    assert prec > 0.7, "Precision is too low (test_logisticRegression_underSampling original data)"
    assert rec > 0.5, "Recall is too low (test_logisticRegression_underSampling original data)"
    assert f1 > 0.1, "F1-Score is too low (test_logisticRegression_underSampling original data)"


#Decision Tree Classifier
def test_decisionTree_underSampling(under_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = under_sampling_data


    #Testing model on under sampled data
    # Train Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_decisionTree_underSampling)"
    assert prec > 0.8, "Precision is too low (test_decisionTree_underSampling)"
    assert rec > 0.8, "Recall is too low (test_decisionTree_underSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_decisionTree_underSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_decisionTree_underSampling original data)"
    assert prec > 0.7, "Precision is too low (test_decisionTree_underSampling original data)"
    assert rec > 0.5, "Recall is too low (test_decisionTree_underSampling original data)"
    assert f1 > 0.1, "F1-Score is too low (test_decisionTree_underSampling original data)"



#Random Forest Classifier
def test_randomForest_underSampling(under_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = under_sampling_data


    #Testing model on under sampled data
    # Train Decision Tree Classifier
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_randomForest_underSampling)"
    assert prec > 0.8, "Precision is too low (test_randomForest_underSampling)"
    assert rec > 0.8, "Recall is too low (test_randomForest_underSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_randomForest_underSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_randomForest_underSampling original data)"
    assert prec > 0.0, "Precision is too low (test_randomForest_underSampling original data)"
    assert rec > 0.7, "Recall is too low (test_randomForest_underSampling original data)"
    assert f1 > 0.1, "F1-Score is too low (test_randomForest_underSampling original data)"

###---------------------------------------------------------------------------
###----------------End of Under Sampling--------------------------------------
###---------------------------------------------------------------------------



###---------------------------------------------------------------------------
###----------------Over Sampling----------------------------------------------
###---------------------------------------------------------------------------
#Logistic Regression
def test_logisticRegression_overSampling(over_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = over_sampling_data


    #Testing model on under sampled data
    # Train Logistc Regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_logisticRegression_overSampling)"
    assert prec > 0.8, "Precision is too low (test_logisticRegression_overSampling)"
    assert rec > 0.8, "Recall is too low (test_logisticRegression_overSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_logisticRegression_overSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_logisticRegression_overSampling origional data)"
    assert prec > 0.0, "Precision is too low (test_logisticRegression_overSampling origional data)"
    assert rec > 0.4, "Recall is too low (test_logisticRegression_overSampling origional data)"
    assert f1 > 0.1, "F1-Score is too low (test_logisticRegression_overSampling origional data)"


#Decision Tree Classifier
def test_decisionTree_overSampling(over_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = over_sampling_data


    #Testing model on under sampled data
    # Train Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_decisionTree_overSampling)"
    assert prec > 0.8, "Precision is too low (test_decisionTree_overSampling)"
    assert rec > 0.8, "Recall is too low (test_decisionTree_overSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_decisionTree_overSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_decisionTree_overSampling original data)"
    assert prec > 0.6, "Precision is too low (test_decisionTree_overSampling original data)"
    assert rec > 0.8, "Recall is too low (test_decisionTree_overSampling original data)"
    assert f1 > 0.7, "F1-Score is too low (test_decisionTree_overSampling original data)"


#Random Forest Classifier
def test_randomForest_overSampling(over_sampling_data, origional_data):
    X_train, X_test, y_train, y_test = over_sampling_data


    #Testing model on under sampled data
    # Train Decision Tree Classifier
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    
    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    y_pred = model.predict(X_test)
    
    # Assert that predictions are not empty
    assert len(y_pred) == len(y_test), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_randomForest_overSampling)"
    assert prec > 0.8, "Precision is too low (test_randomForest_overSampling)"
    assert rec > 0.8, "Recall is too low (test_randomForest_overSampling)"
    assert f1 > 0.8, "F1-Score is too low (test_randomForest_overSampling)"

    
    #Testing model on original data
    org_X_train, org_X_test, org_Y_train, org_Y_test = origional_data

    model.fit(org_X_train,org_Y_train)

    # Assert the model has been trained
    assert model is not None, "Model training failed"
    
    org_y_pred = model.predict(org_X_test)
    
    # Assert that predictions are not empty
    assert len(org_Y_test) == len(org_y_pred), "Prediction length mismatch"

    # Calculate metrics
    acc = accuracy_score(org_Y_test, org_y_pred)
    prec = precision_score(org_Y_test, org_y_pred)
    rec = recall_score(org_Y_test, org_y_pred)
    f1 = f1_score(org_Y_test, org_y_pred)
    
    # Assert accuracy and other metrics are within expected ranges
    assert acc > 0.8, "Accuracy is too low (test_randomForest_overSampling original data)"
    assert prec > 0.8, "Precision is too low (test_randomForest_overSampling original data)"
    assert rec > 0.7, "Recall is too low (test_randomForest_overSampling original data)"
    assert f1 > 0.7, "F1-Score is too low (test_randomForest_overSampling original data)"

###---------------------------------------------------------------------------
###----------------End of Over Sampling---------------------------------------
###---------------------------------------------------------------------------