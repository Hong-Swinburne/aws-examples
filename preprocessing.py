# processing data
import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Here we have defined all the columns that are present in our data
columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content',
'Item_Visibility','Item_Type', 'Item_MRP', 'Outlet_Identifier',
'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
'Outlet_Type', 'Item_Outlet_Sales']

# This method will help us in printing the shape of our data
def print_shape(df):
    print('Data shape: {}'.format(df.shape))
    
if __name__=='__main__':
    # At the time of container execution we will use this parser to define our train test split. Default kept is 10%
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.1)
    parser.add_argument('--train-val-split-ratio', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    # This is the data path inside the container where the Train.csv will be downloaded and saved
    input_data_path = os.path.join('/opt/ml/processing/input', 'Train.csv')
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path)
    data = pd.DataFrame(data=data, columns=columns)
    
    for i in data.Item_Type.value_counts().index:
        data.loc[(data['Item_Weight'].isna()) & (data['Item_Type'] == i),['Item_Weight']] = \
            data.loc[data['Item_Type'] == 'Fruits and Vegetables', ['Item_Weight']].mean()[0]
            
    cat_data = data.select_dtypes(object)
    num_data = data.select_dtypes(np.number)
    
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Grocery Store'), ['Outlet_Size']] = 'Small'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type1'), ['Outlet_Size']] = 'Small'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type2'), ['Outlet_Size']] = 'Medium'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type3'), ['Outlet_Size']] = 'Medium'
    cat_data.loc[cat_data['Item_Fat_Content'] == 'LF' , ['Item_Fat_Content']] = 'Low Fat'
    cat_data.loc[cat_data['Item_Fat_Content'] == 'reg' , ['Item_Fat_Content']] = 'Regular'
    cat_data.loc[cat_data['Item_Fat_Content'] == 'low fat' , ['Item_Fat_Content']] = 'Low Fat'
    
    le = LabelEncoder()
    cat_data = cat_data.apply(le.fit_transform)
    ss = StandardScaler()
    num_data = pd.DataFrame(ss.fit_transform(num_data), columns = num_data.columns)
    cat_data = pd.DataFrame(ss.fit_transform(cat_data), columns = cat_data.columns)
    final_data = pd.concat([num_data,cat_data],axis=1)
    print('Data after cleaning: {}'.format(final_data.shape))
    
       
    X = final_data.drop(['Item_Outlet_Sales'], axis=1)
    y = data['Item_Outlet_Sales']
    
    # prepare categorical labels for sales classification
    # bin the Sales columns into four categories for classification problem
    y_binned = pd.cut(data['Item_Outlet_Sales'], 4, labels=['A', 'B', 'C', 'D'])
    # label encoding of y_binned
    temp = le.fit(y_binned)
    y_clf = temp.transform(y_binned)
    
    # prepare dataset for training
    train_test_ratio = args.train_test_split_ratio
    train_val_ratio = args.train_val_split_ratio
    
    print('Splitting data into train and test sets with ratio {}'.format(train_test_ratio))
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y, test_size=train_test_ratio, random_state=0)
    
    # This defines the output path inside the container from where all the csv sheets will be taken and uploaded to S3 Bucket
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_reg_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_reg_labels.csv')
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_reg_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_reg_labels.csv')
    
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(X_test).to_csv(test_features_output_path, header=False, index=False)
    print('Saving training labels to {}'.format(train_labels_output_path))
    y_reg_train.to_csv(train_labels_output_path, header=False, index=False)
    print('Saving test labels to {}'.format(test_labels_output_path))
    y_reg_test.to_csv(test_labels_output_path, header=False, index=False)
    
    # prepare data for classification
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=train_test_ratio, random_state=0)
    print('Splitting data into train and validation sets with ratio {}'.format(train_val_ratio))
    X_train, X_val, y_clf_train, y_clf_val = train_test_split(X_train, y_clf_train, test_size=train_val_ratio, random_state=0)
    
    # form training/val/test dataset by concatenating the numeric features with the true labels
    train_clf = pd.concat([pd.Series(y_clf_train, index=X_train.index, name='Item_Outlet_Sales', dtype=int), X_train], axis=1)
    val_clf = pd.concat([pd.Series(y_clf_val, index=X_val.index, name='Item_Outlet_Sales', dtype=int), X_val], axis=1)
    test_clf = pd.concat([pd.Series(y_clf_test, index=X_test.index, name='Item_Outlet_Sales', dtype=int), X_test], axis=1)
    
    # This defines the output path inside the container from where all the csv sheets will be taken and uploaded to S3 Bucket
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_clf_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_clf_labels.csv')
    train_output_path = os.path.join('/opt/ml/processing/train', 'train_clf.csv')
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_clf_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_clf_labels.csv')
    test_output_path = os.path.join('/opt/ml/processing/test', 'test_clf.csv')
    val_features_output_path = os.path.join('/opt/ml/processing/val', 'val_clf_features.csv')
    val_labels_output_path = os.path.join('/opt/ml/processing/val', 'val_clf_labels.csv')
    val_output_path = os.path.join('/opt/ml/processing/val', 'val_clf.csv')
    
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)
    print('Saving val features to {}'.format(val_features_output_path))
    pd.DataFrame(X_val).to_csv(val_features_output_path, header=False, index=False)
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(X_test).to_csv(test_features_output_path, header=False, index=False)
    print('Saving training labels to {}'.format(train_labels_output_path))
    pd.DataFrame(y_clf_train).to_csv(train_labels_output_path, header=False, index=False)
    print('Saving val labels to {}'.format(val_labels_output_path))
    pd.DataFrame(y_clf_val).to_csv(val_labels_output_path, header=False, index=False)
    print('Saving test labels to {}'.format(test_labels_output_path))
    pd.DataFrame(y_clf_test).to_csv(test_labels_output_path, header=False, index=False)
    print('Saving training data to {}'.format(train_output_path))
    pd.DataFrame(train_clf).to_csv(train_output_path, header=False, index=False)
    print('Saving val data to {}'.format(val_output_path))
    pd.DataFrame(val_clf).to_csv(val_output_path, header=False, index=False)
    print('Saving test data to {}'.format(test_output_path))
    pd.DataFrame(test_clf).to_csv(test_output_path, header=False, index=False)