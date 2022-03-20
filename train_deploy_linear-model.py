# training a linear model for prediction on big-mart sales data into 4 categories
import os, io, boto3, sagemaker
import pandas as pd
import numpy as np
from sagemaker.amazon.amazon_estimator import get_image_uri
import sagemaker.amazon.common as smac

# role = sagemaker.get_execution_role()
# manually set the ARN for AWS IAM role. The ARN can be found when you create an IAM role for executing SageMaker
role = 'arn:aws:iam::842077957268:role/service-role/AmazonSageMaker-ExecutionRole-20220305T165318'


# obtain training data in S3 bucket
train_features = "s3://slytherins-test/train_clf_features.csv"
X = pd.read_csv(train_features)
train_labels = "s3://slytherins-test/train_clf_labels.csv"
y = pd.read_csv(train_labels)

# convert data to RecordIO-Protobuf format for training model in streaming data
vectors = np.array(X.values, dtype='float32')
labels = np.array(y, dtype='float32').reshape(-1, )
print("vector shape:{}, label shape:{}".format(vectors.shape, labels.shape))
# convert data into RecordIO-Protobuf format with small chunks and store chunks in a temporary buffer
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)

# upload buffer data to S3 and close the buffer
key = 'recordio-pb-data'
bucket = 'slytherins-test'
prefix = 'linear-learner'
file_path = os.path.join(prefix, 'train', key)
# upload data
boto3.resource('s3').Bucket(bucket).Object(file_path).upload_fileobj(buf)
# Obtain the URL of data in S3 bucket for access
s3_train_data = 's3://{}/{}'.format(bucket, file_path)
print('uploaded training data location: {}'.format(s3_train_data))

# obtain the built-in linear model container
container = get_image_uri(boto3.Session().region_name, 'linear-learner')
# pass the required parameters for linear learner and initialize the algorithm
sess = sagemaker.Session()
# S3 location for saving the training result (model artifacts and output files)
output_location = "s3://slytherins-test/linear-learner/clf_model" 
linear = sagemaker.estimator.Estimator(container,
                                        role,
                                        train_instance_count=1,
                                        train_instance_type='ml.m4.xlarge',
                                        output_path=output_location,
                                        sagemaker_session=sess)
# set up hyperparameters for classification problem
linear.set_hyperparameters(feature_dim=11,
                            predictor_type='multiclass_classifier',
                            mini_batch_size=100,
                            num_classes=4)

# set up hyperparameters for regression problem
# linear.set_hyperparameters(feature_dim=11,
                            # predictor_type='regression',
                            # mini_batch_size=100)

# start training
print("start training linear model")
linear.fit({'train': s3_train_data})

# deploy the linear model
linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# prepare for test data
test_features = "s3://slytherins-test/test_clf_features.csv"
test_vectors = pd.read_csv(test_features)
test_labels = "s3://slytherins-test/test_clf_labels.csv"
test_labels = pd.read_csv(test_labels)
test_vectors = np.array(test_vectors.values, dtype='float32')
test_labels = np.array(test_labels, dtype='float32').reshape(-1, )
print("vector shape:{}, label shape:{}".format(test_vectors.shape, test_labels.shape))

print("finished test data preparation")

# model prediction
from sagemaker.predictor import csv_serializer, json_deserializer
# linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer
# predict a single example
result = linear_predictor.predict(test_vectors[0])
print(result)

# predict for multi-examples
predictions = []
for array in np.array_split(test_vectors, 100):
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]
predictions = np.array(predictions)

# evaluate the prediction performance
from sklearn.metrics import precision_score, recall_score, f1_score
print('precision={}'.format(precision_score(test_labels, predictions, average='weighted')))
print('recall={}'.format(recall_score(test_labels, predictions, average='weighted')))
print('f1={}'.format(f1_score(test_labels, predictions, average='weighted')))

# stop the model endpoint
sagemaker.Session().delete_endpoint(linear_predictor.endpoint)
print("model endpoints closed")