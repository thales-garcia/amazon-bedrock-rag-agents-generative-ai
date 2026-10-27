import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")

MODEL_ID = "amazon.nova-lite-v1:0"

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")

s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION)

# for bucket in s3.buckets.all():
#   print(bucket.name)

bedrock = boto3.client(service_name='bedrock',
                       aws_access_key_id=AWS_ACCESS_KEY_ID,
                       aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                       region_name=AWS_REGION)

foundation_models = bedrock.list_foundation_models()
# print(foundation_models)

bedrock_runtime = boto3.client(service_name='bedrock-runtime',
                               aws_access_key_id=AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                               region_name=AWS_BEDROCK_REGION)


def generate_response(prompt, temperature=0.7, top_p=0.9, max_tokens=300):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
    }

    response = bedrock_runtime.invoke_model(modelId="amazon.nova-lite-v1:0", body=json.dumps(body))
    response_body = json.loads(response["body"].read())
    return response_body["output"]["message"]["content"][0]["text"]


prompt = "What is machine learning?"

print(generate_response(prompt))

