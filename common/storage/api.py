import asyncio
import logging
import os
from io import BytesIO
from typing import List

from aiobotocore.session import get_session
from botocore.exceptions import ClientError

BUCKET_NAME = 'clrn-data'


class FileNotFound(RuntimeError):
    pass


class Storage:
    DEFAULT_BASE_DIR = 'data/cloud'

    def __init__(self):
        self._session = get_session()

    def _get_client(self):
        return self._session.create_client('s3',
                                           aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                           aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                                           endpoint_url=os.environ['AWS_ENDPOINT'],
                                           use_ssl=int(os.getenv('AWS_USE_SSL', '0')))

    async def sync_folder(self, remote_folder: str):
        if remote_folder.startswith(self.DEFAULT_BASE_DIR):
            remote_folder = remote_folder.replace(self.DEFAULT_BASE_DIR + '/', '')

        if not remote_folder.endswith('/'):
            remote_folder += '/'
        local_folder = os.path.join(self.DEFAULT_BASE_DIR, remote_folder)

        # download first
        dirs = await self.list_directories(remote_folder)
        jobs = list(map(self.sync_folder, dirs))

        remote_files = await self.list_files(remote_folder)
        existing_files = set()
        for remote_path in remote_files:
            local_path = os.path.join(self.DEFAULT_BASE_DIR, remote_path)

            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                jobs.append(self.download(remote_path, local_path))

            existing_files.add(local_path)

        # now upload
        for path, directories, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(path, file)
                if local_path not in existing_files:
                    jobs.append(self.upload(local_path, local_path.replace(self.DEFAULT_BASE_DIR + '/', '')))

        await asyncio.gather(*jobs)

    async def sync(self, folders: List[str]):
        await asyncio.gather(*map(self.sync_folder, folders))

    async def download(self, file_id: str, result_file):
        if result_file is None:
            result_file = file_id

        logging.info(f"Downloading file: {os.environ['AWS_ENDPOINT']}/{BUCKET_NAME}/{file_id} to {result_file}")

        if file_id.startswith(self.DEFAULT_BASE_DIR):
            file_id = file_id.replace(self.DEFAULT_BASE_DIR + '/', '')

        async with self._get_client() as client:
            try:
                if hasattr(result_file, 'write'):
                    response = await client.get_object(Bucket=BUCKET_NAME, Key=file_id)
                    async with response['Body'] as stream:
                        while True:
                            chunk = await stream.read(8192)
                            if not chunk:
                                break
                            result_file.write(chunk.decode())
                else:
                    response = await client.get_object(Bucket=BUCKET_NAME, Key=file_id)
                    async with response['Body'] as stream:
                        os.makedirs(os.path.dirname(result_file), exist_ok=True)
                        with open(result_file, 'wb') as file_data:
                            while True:
                                chunk = await stream.read(8192)
                                if not chunk:
                                    break
                                file_data.write(chunk)
            except ClientError as ex:
                if ex.response['Error']['Code'] == 'NoSuchKey':
                    logging.warning(f"File: {os.environ['AWS_ENDPOINT']}/{BUCKET_NAME}/{file_id} not found")
                    raise FileNotFound(file_id)
                else:
                    raise

    async def delete(self, file_id):
        logging.info(f"Deleting file: {file_id} from {os.environ['AWS_ENDPOINT']}")
        if file_id.startswith(self.DEFAULT_BASE_DIR):
            file_id = file_id.replace(self.DEFAULT_BASE_DIR + '/', '')

        async with self._get_client() as client:
            await client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': [{'Key': file_id, }]})

    async def upload(self, file_object, file_id: str):
        if file_id.startswith(self.DEFAULT_BASE_DIR):
            file_id = file_id.replace(self.DEFAULT_BASE_DIR + '/', '')

        logging.info(f"Uploading file: {file_id} to {os.environ['AWS_ENDPOINT']}")

        async with self._get_client() as client:
            if isinstance(file_object, bytes):
                await client.put_object(Bucket=BUCKET_NAME, Key=file_id, Body=BytesIO(file_object))
            elif hasattr(file_object, 'read'):
                await client.put_object(Bucket=BUCKET_NAME, Key=file_id, Body=file_object)
            else:
                await client.put_object(Bucket=BUCKET_NAME, Key=file_id, Body=open(file_object, 'rb'))

    async def list_directories(self, path: str):
        if path.startswith(self.DEFAULT_BASE_DIR):
            path = path.replace(self.DEFAULT_BASE_DIR + '/', '')

        res = list()
        async with self._get_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            async for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=path, Delimiter='/'):
                for prefix in result.get('CommonPrefixes', []):
                    res.append(prefix['Prefix'])

        return res

    async def list_files(self, path: str):
        if path.startswith(self.DEFAULT_BASE_DIR):
            path = path.replace(self.DEFAULT_BASE_DIR + '/', '')

        res = list()
        async with self._get_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            async for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=path, Delimiter='/'):
                for key in result.get('Contents', []):
                    res.append(key['Key'])

        return res

    async def sign(self, path: str):
        async with self._get_client() as client:
            return await client.generate_presigned_url('get_object',
                                                       Params={'Bucket': BUCKET_NAME, 'Key': path})


if __name__ == '__main__':
    import sys

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)

    storage = Storage()
    operation = getattr(storage, sys.argv[1])
    asyncio.get_event_loop().run_until_complete(operation(*sys.argv[2:]))
