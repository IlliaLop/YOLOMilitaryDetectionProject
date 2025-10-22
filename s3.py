import boto3
import uuid
import os

def upload_and_get_temporary_url(file_path: str, bucket_name: str, expiration_in_seconds: int = 3600):

    s3 = boto3.client("s3")
    file_extension = os.path.splitext(file_path)[1]
    unique_key = str(uuid.uuid4()) + file_extension

    s3.upload_file(file_path, bucket_name, unique_key, ExtraArgs={"ContentType": "image/jpeg"})

    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket_name,
            "Key": unique_key
        },
        ExpiresIn=expiration_in_seconds
    )
    return presigned_url, unique_key