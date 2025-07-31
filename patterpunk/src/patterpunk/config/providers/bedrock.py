import os
from typing import Optional, Union, Literal

AWS_REGION = os.getenv("PP_AWS_REGION", None)
AWS_ACCESS_KEY_ID = os.getenv("PP_AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("PP_AWS_SECRET_ACCESS_KEY", None)

_boto3 = None


def get_boto3():
    global _boto3
    if _boto3 is None and AWS_REGION:
        import boto3

        _boto3 = boto3
    return _boto3


def get_bedrock_client_by_region(
    client_type: Union[Literal["bedrock"], Literal["bedrock-runtime"]],
    region: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
):
    boto3_module = get_boto3()
    if not boto3_module:
        return None

    if aws_access_key_id and aws_secret_access_key:
        access_key_id = aws_access_key_id
        secret_access_key = aws_secret_access_key
    else:
        access_key_id = AWS_ACCESS_KEY_ID
        secret_access_key = AWS_SECRET_ACCESS_KEY

    if region is None:
        region = AWS_REGION

    if access_key_id and secret_access_key:
        return boto3_module.client(
            client_type,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
    else:
        return boto3_module.client(
            client_type,
            region_name=region,
        )


def is_bedrock_available() -> bool:
    return AWS_REGION is not None


boto3 = get_boto3()
