# This script performs the processing jobs for custom container created by ScriptProcessor
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role
# role = get_execution_role()
# manually set the ARN for AWS IAM role. The ARN can be found when you create an IAM role for executing SageMaker
role = 'arn:aws:iam::842077957268:role/service-role/AmazonSageMaker-ExecutionRole-20220305T165318'

import boto3
account_id = boto3.client('sts').get_caller_identity().get('Account')

# manually set the account_id for AWS account
# account_id = "842077957268"

ecr_repository = 'big-mart-sales-sagemaker-processing-container'
tag = ':latest'
region = boto3.session.Session().region_name

# Define the ECR repository address and execute the command-line scripts to build and push the image to AWS ECR
processing_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account_id, region, ecr_repository + tag)

# run preprocessing job
script_processor = ScriptProcessor(command=['python3'],
                                    image_uri=processing_repository_uri,
                                    role=role,
                                    instance_count=1,
                                    instance_type='ml.m5.xlarge')

input_data = 's3://slytherins-test/Train.csv'
script_processor.run(code='preprocessing.py',
                    inputs=[ProcessingInput(source=input_data, destination='/opt/ml/processing/input')],
                    outputs=[ProcessingOutput(source='/opt/ml/processing/train', destination='s3://slytherins-test/'),
                              ProcessingOutput(source='/opt/ml/processing/test', destination='s3://slytherins-test/')])

print('finish the preprocessing job')

# get detailed information about the job
preprocessing_job_description = script_processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'output-1':
        preprocessed_training_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'output-2':
        preprocessed_test_data = output['S3Output']['S3Uri']

# check the output by reading the data using Pandas.
import pandas as pd
training_features = pd.read_csv(preprocessed_training_data + 'train_features.csv', nrows=10, header=None)
print('Training features shape: {}'.format(training_features.shape))
print(training_features.head(n=10))