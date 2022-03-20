# training a xgboost model for prediction on big-mart sales data into 4 categories
import os, boto3, sagemaker
import pandas as pd
import numpy as np

# manually set the ARN for AWS IAM role. The ARN can be found when you create an IAM role for executing SageMaker
role = 'arn:aws:iam::842077957268:role/service-role/AmazonSageMaker-ExecutionRole-20220305T165318'
region = boto3.Session().region_name
bucket = 'slytherins-test'
prefix = 'xgboost'

from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput

# URL of S3 bucket for storing the trained model
s3_output_location='s3://{}/{}/{}'.format(bucket, prefix, 'xgboost_model')

# download xgboost container
container=sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
print(container)

# build xgboost model
xgb_model=sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5, # allocate 5GB storage volume to attach to the training instance
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    # Specify a list of SageMaker Debugger built-in rules,
    # e.g.creates an XGBoost report that provides insights into the training progress and results
    rules=[Rule.sagemaker(rule_configs.create_xgboost_report())]
)

# set up model parameters
xgb_model.set_hyperparameters(
    max_depth = 5,
    eta = 0.2,
    gamma = 4,
    min_child_weight = 6,
    subsample = 0.7,
    objective = "multi:softmax",
    num_round = 50,
    num_class = 4
)

# configure data input flow for training
from sagemaker.session import TrainingInput

train_input = TrainingInput(
    "s3://{}/{}".format(bucket, "train_clf.csv"), content_type="csv"
)
validation_input = TrainingInput(
    "s3://{}/{}".format(bucket, "val_clf.csv"), content_type="csv"
)

# start training
print("\nstart training...\n")
xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)

# deploy the model
from sagemaker.serializers import CSVSerializer
xgb_predictor=xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
)
print(xgb_predictor.endpoint_name)

"""
After you deploy the model to an endpoint, you can set up a new SageMaker predictor by pairing the 
endpoint and continuously make real-time predictions in any other notebooks. The following example 
code shows how to use the SageMaker Predictor class to set up a new predictor object using the same 
endpoint. Re-use the endpoint name that you used for the xgb_predictor
"""
# use the deployed model for real-time prediction
xgb_predictor_reuse=sagemaker.predictor.Predictor(
    endpoint_name=xgb_predictor.endpoint_name,
    sagemaker_session=sagemaker.Session(),
    serializer=sagemaker.serializers.CSVSerializer()
)

# prepare test samples
test_features = "s3://slytherins-test/test_clf_features.csv"
test_vectors = pd.read_csv(test_features)
test_vectors = np.array(test_vectors.values, dtype='float32')
# predict a single example
result = xgb_predictor_reuse.predict(test_vectors[0])
print("\n result for test example[0]:{}".format(result))

# close the endpoint
client = boto3.client('sagemaker', region_name = region)
client.delete_endpoint(
    EndpointName=xgb_predictor.endpoint_name,
)
print("endpoint closed")

# predict a batch of examples using batch transform
print("start to perform batch prediction\n")
# The location of the test dataset
batch_input = "s3://{}/{}".format(bucket, "test_clf_features.csv")
# The location to store the results of the batch transform job
batch_output = 's3://{}/{}/batch-prediction'.format(bucket, prefix)

# Create a transformer object
transformer = xgb_model.transformer(
    instance_count=1, 
    instance_type='ml.m4.xlarge', 
    output_path=batch_output
)

# Initiate the batch transform job
transformer.transform(
    data=batch_input, 
    data_type='S3Prefix',
    content_type='text/csv', 
    split_type='Line'
)
transformer.wait()
