# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "botocore",
# ]
# ///
"""post process script for training at GPU server. called by run.sh.

This script downloads files from S3 and creates data.yml from data_template.yml.
you should run this script before training at GPU server (automatically called by run.sh).
see also: run.sh, train.py
"""

import json
from pathlib import Path

import boto3
import tomllib
from botocore.client import Config


def download_files_from_s3(uris: dict[str, str], access_key: str, secret_key: str, endpoint_url: str):
    # Initialize S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        config=Config(signature_version="s3v4"),
    )

    # Ensure the download directory exists
    for s3_uri, real_path in uris.items():
        if Path(real_path).exists():
            continue

        # S3 URI parsing
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}. It must start with s3://")

        s3_uri_parts = s3_uri[5:].split("/", 1)
        bucket_name = s3_uri_parts[0]
        prefix = s3_uri_parts[1] if len(s3_uri_parts) > 1 else ""
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if "Contents" not in response:
            print(f"No files found for the given S3 URI: {s3_uri}.")
            continue

        file_key = response["Contents"][0]["Key"]
        print(f"Downloading {file_key} to {real_path}...")
        s3.download_file(Bucket=bucket_name, Key=file_key, Filename=real_path)

    print("Download completed.")


def main():
    # Replace with your values
    with open("env.toml") as f:
        env = tomllib.load(f)

    # Load download info
    with open("download_path.json") as f:
        data: dict[str, str] = json.load(f)

    # Download files from S3
    download_files_from_s3(data, env["access_key"], env["secret_key"], env["endpoint_url"])

    # Create data.yml from template
    with open("data_template.yml", encoding="utf8") as rf, open("data.yml", mode="w", encoding="utf8") as wf:
        wf.write(rf.read().replace("__file__", Path(__file__).resolve().parent.joinpath("datasets").as_posix()))


if __name__ == "__main__":
    main()
