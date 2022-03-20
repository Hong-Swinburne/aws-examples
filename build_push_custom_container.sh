#!/bin/sh
ecr_repository="big-mart-sales-sagemaker-processing-container"
tag=":latest"
ecr_repository_tag="big-mart-sales-sagemaker-processing-container:latest"

account_id="842077957268"
region="ap-southeast-2"
processing_repository_uri="842077957268.dkr.ecr.ap-southeast-2.amazonaws.com/big-mart-sales-sagemaker-processing-container:latest"

# Builds the image
docker build -t $ecr_repository docker
# Logs in to AWS
$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
# Creates ECR Repository
aws ecr create-repository --repository-name $ecr_repository
# Tags the image to differentiate it from other images
docker tag $ecr_repository_tag $processing_repository_uri
# Pushes image to ECR
docker push $processing_repository_uri
