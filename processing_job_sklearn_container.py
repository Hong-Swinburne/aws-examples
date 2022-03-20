# create the AWS built-in sklearn container to perform the preprocessing job

import boto3
import sagemaker
from sagemaker import get_execution_role
region = boto3.session.Session().region_name
# role = get_execution_role()
# manually set the ARN for AWS IAM role. The ARN can be found when you create an IAM role for executing SageMaker
role = 'arn:aws:iam::842077957268:role/service-role/AmazonSageMaker-ExecutionRole-20220305T165318'

# create sklearn container in AWS
from sagemaker.sklearn.processing import SKLearnProcessor
sklearn_processor = SKLearnProcessor(framework_version='0.20.0', role=role, instance_type='ml.m5.xlarge', instance_count=1)

# load data
import pandas as pd
input_data = 's3://slytherins-test/Train.csv'
# input_data = 'dataset/sales/Train.csv'
df = pd.read_csv(input_data)
print(df.head())

# run preprocessing job
from sagemaker.processing import ProcessingInput, ProcessingOutput
sklearn_processor.run(code='preprocessing.py',
                      inputs = [ProcessingInput(source=input_data, 
                                                destination='/opt/ml/processing/input')],
                      outputs=[ProcessingOutput(output_name='train_data', 
                                                source='/opt/ml/processing/train',
                                                destination='s3://slytherins-test/'),
                               ProcessingOutput(output_name='val_data', 
                                                source='/opt/ml/processing/val',
                                                destination='s3://slytherins-test/'),
                              ProcessingOutput(output_name='test_data',
                                               source='/opt/ml/processing/test',
                                               destination='s3://slytherins-test/')],
                    arguments=['--train-test-split-ratio', '0.1', '--train-val-split-ratio', '0.2']
                    )

print('finish the preprocessing job')

# get detailed information about the job
preprocessing_job_description = sklearn_processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'train_data':
        preprocessed_training_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'val_data':
        preprocessed_val_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'test_data':
        preprocessed_test_data = output['S3Output']['S3Uri']

# check the output by reading the data using Pandas.
training_features = pd.read_csv(preprocessed_training_data + 'train_clf_features.csv', nrows=10, header=None)
print('Training features shape: {}'.format(training_features.shape))
print(training_features.head(10))