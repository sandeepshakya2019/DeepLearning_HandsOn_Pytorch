
# import download_package
import requests
import os
import zipfile

# Define the download URL and destination paths
url = "https://storage.googleapis.com/kaggle-data-sets/6263674/10147053/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250410%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250410T084130Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5ce3e680acf4a64cf5fadc69058717d0f8fe0f09f2e8222107a681ee751e2131c7d9059e52d7a3ac8ce7c366c329e6dd8f38ea9bf2edbf20cf3f8b1b8384aedeaa5d4182f790a672b74530f0d43c420bb1c429ddb7d24c96253588e0c8888835dc2a0d57331666d4fed6d48365c948a2480049e4b5d663d0a284c25a4af36d54450675616bfc9b752c58783028be712ce9229e443fe5e670f2d602b609d12005832e023e2c486188f78d3e0acdd9a0d6b325c5311a9d27173d1674f008e4ebb5e214ba2188ff01cef0c6e454dfc932391082d857cde2c71c4d9e99f57fce9a9b1266e78a019b5421ebfac4f2da15680aac321f2a3427bdc02ffc01f64a1acd06"
download_path = "archive.zip"
extract_path = "dataset"

# Download the dataset
print("[+] Downloading dataset...")
response = requests.get(url, stream=True)
with open(download_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print("[✓] Download complete.")

# Extract the dataset
print("[+] Extracting zip file...")
with zipfile.ZipFile(download_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("[✓] Extraction complete.")

# Show extracted path
print("Dataset extracted to:", os.path.abspath(extract_path))
