import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def clean_features(df, features):
    for feature in features:
        # strip the '$' sign
        df[feature] = df[feature].str.strip('$')
        # remove the ',' sign
        df[feature] = df[feature].str.replace(',', '')
        # convert to float
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    return df


def process_missing(df, missing_entries):
    # use the 'mean' strategy
    imputer = SimpleImputer(strategy="mean")
    df_copy = df[missing_entries]
    # fit and transform the dataset with newly filled entries
    df_copy = imputer.fit_transform(df_copy)
    df_copy = pd.DataFrame(df_copy)
    # set the original dataset columns to newly filled columns
    df["CAR_AGE"] = df_copy[0]
    df["HOME_VAL"] = df_copy[1]
    df["INCOME"] = df_copy[2]
    df["YOJ"] = df_copy[3]
    return df


def process_dates(df):
    # change the birth column into an appropriate format
    df['newbirth'] = df['BIRTH'].str[:2] + '-' + \
        df['BIRTH'].str[2:5] + '-' + '19' + df['BIRTH'].str[5:7]
    # convert it to datetime object
    df['newbirth'] = pd.to_datetime(df['newbirth'])
    # calculate the age
    df['AGE'] = pd.to_datetime('today') - df['newbirth']
    # calculate the number of years
    df['AGE'] = df['AGE'] / np.timedelta64(1, 'Y')
    # round up the ages
    df['AGE'] = df['AGE'].apply(np.ceil)
    # drop the `BIRTH` column
    df.drop('BIRTH', axis=1, inplace=True)
    # drop the temporary column that was created
    df.drop('newbirth', axis=1, inplace=True)
    return df


def drop_missing(df, missing_columns):
    df = df.dropna(subset=['OCCUPATION'])
    return df


def encode_features(df, ordinal_features, nominal_features):
    label = LabelEncoder()

    for ordinal_feature in ordinal_features:
        # fit and transform every ordinal feature in its encoded form
        df[ordinal_feature] = label.fit_transform(df[ordinal_feature])

    for nominal_feature in nominal_features:
        # create dummy columns for every nominal feature
        encoded = pd.get_dummies(df[nominal_feature])
        # drop the old nominal feature column
        df = df.drop(nominal_feature, axis=1)
        # add the new encoded columns
        df = df.join(encoded)

    return df


def pre_modeling(df):
    
    X = df.drop('CLM_AMT', axis = 1)
    y = df["CLM_AMT"]
   
    return X, y


def model(X_train, y_train):
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X, scaler):
    #scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X)

    y_pred = clf.predict(X_test)
 
    amount = scaler.inverse_transform(y_pred)
    return y_pred
    
    

def main():
    df = pd.read_csv('car_insurance_claim.csv')
    features = ["HOME_VAL", "INCOME", "CLM_AMT", "BLUEBOOK", "OLDCLAIM"]
    df = clean_features(df, features)
    missing_entries = ["CAR_AGE", "HOME_VAL", "INCOME", "YOJ"]
    df = process_missing(df, missing_entries)
    df = process_dates(df)
    missing_columns = ['OCCUPATION']
    df = drop_missing(df, missing_columns)

    # ordinal_features = ["PARENT1", "MSTATUS", "GENDER",
    # "EDUCATION", "CAR_USE", "RED_CAR", "REVOKED"]
    #nominal_features = ["CAR_TYPE", "OCCUPATION", "URBANICITY"]
    #df = encode_features(df, ordinal_features, nominal_features)
    
    df = df[df['CLAIM_FLAG'] == 1]
    
    df = df[['AGE', 'INCOME', 'BLUEBOOK', 'OLDCLAIM', 'TRAVTIME',
             'MVR_PTS', 'YOJ', 'CAR_AGE', 'TIF', 'CLM_FREQ', 'CLM_AMT']]
    
    X, y = pre_modeling(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 
                                                        y, # dependent variable
                                                        test_size = 0.2) # percentage of data to use for test set
    
    scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    
    y_train_arr = y_train.values
    
    reshaped_y_train = y_train_arr.reshape(-1, 1)
    
    y_train = scaler.fit_transform(reshaped_y_train)
        
    
    clf = model(X_train, y_train)
    pickle.dump((clf, scaler), open('model11.pkl', 'wb'))


main()


