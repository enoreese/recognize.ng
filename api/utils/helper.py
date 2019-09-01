import os, json
from datetime import datetime
from cloudinary.uploader import upload
from botocore.exceptions import ClientError
from api.core import logger
import boto3, cloudinary


def cloudinary_upload(file, filename, bucket):
    public_id = 'recognize/{}/{}'.format(bucket, filename)
    upload_result = upload(file=file, public_id=public_id)
    url = upload_result['url']

    return url


def aws_upload(file,
               filename,
               bucket,
               object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = filename

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True


def local_upload(file, filepath):
    # save file
    try:
        file.save(filepath)
        return filepath
    except:
        return False


def handle_upload(file, storage='local', data=None, bucket='uploads', prefix=None):
    if not file:
        return 'no file provided.'

    if storage == 'local':
        directory = 'api/uploads/{}'.format(data['user_id'])
        filename = str(prefix) + '-' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        # create directory if it doesnt exist
        os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory, filename)

        return local_upload(file=file, filepath=filepath)

    if storage == 'cloudinary':
        cloudinary.config(
            api_key=os.environ['CLOUDNIARY_API_KEY'],
            api_secret=os.environ['CLOUDNINARY_API_SECRET'],
            cloud_name=os.environ['CLOUDINARY_CLOUD_NAME']
        )
        filename = str(prefix) + '-' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        return cloudinary_upload(file=file, filename=filename, bucket=bucket)

    if storage == 'aws':
        filename = str(prefix) + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        if aws_upload(file=file, filename=filename, bucket=bucket):
            return filename
        else:
            return False


def rename_file(storage, bucket, old_filename, new_filename):
    if storage == 'local':
        base = 'api/uploads/{}'.format(bucket)
        old_directory = os.path.join(base, old_filename)
        new_directory = os.path.join(base, new_filename)
        os.rename(old_directory, new_directory)
    if storage == 'aws':
        s3 = boto3.resource('s3')
        source = bucket + old_filename
        s3.Object(bucket, new_filename).copy_from(CopySource=source)
        s3.Object(bucket, old_filename).delete()
    if storage == 'cloudinary':
        result = cloudinary.uploader.rename(old_filename, new_filename)