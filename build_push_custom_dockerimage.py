# This script builds and pushes custom docker image for preprocessing data using customised container with ScriptProcessor class
import boto3
account_id = boto3.client('sts').get_caller_identity().get('Account')

# manually set the account_id for AWS account
# account_id = "842077957268"

ecr_repository = 'big-mart-sales-sagemaker-processing-container'
tag = ':latest'
region = boto3.session.Session().region_name

# Define the ECR repository address and execute the command-line scripts to build and push the image to AWS ECR
processing_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account_id, region, ecr_repository + tag)
print(processing_repository_uri)

# Commands for building and pushing Docker image to ECR
# Create ECR repository and push docker image
import os, subprocess
# # Builds the image
# print(subprocess.call(["docker", "build", "-t $ecr_repository", "docker"], shell=False))
# # Logs in to AWS
# print(os.popen("$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)").read()) 
# # Creates ECR Repository
# print(subprocess.call(["aws ecr create-repository", "--repository-name $ecr_repository"], shell=False))
# # Tags the image to differentiate it from other images
# print(subprocess.call(["docker tag", "{ecr_repository + tag}", "$processing_repository_uri"], shell=False))
# # Pushes image to ECR
# print(subprocess.call(["docker push", "$processing_repository_uri"], shell=False))

# run the shell script to build and push docker container
print(os.popen("./build_push_custom_container.sh").read())