import urllib.request
import zipfile
import os

# set the download URL and destination file path
url = "https://p-lux1.pcloud.com/cBZFsIjWMZkN4hxdZZZdDnjo7Z2ZZKs0ZkZC3n1jZn4ZVHZMFZ6RZZOXZ4RZKLZKpZazZrzZQzZwFZuLZ6DBkVZX94i6UOHcVJkS6BArn43VQISbtxV/deeplab_upernet_aspp_psp_ab_swin_skips_1288_0.0003.zip"
dest_file_path = "model_weights.zip"

# download the file
print("Downloading model weights...")
# urllib.request.urlretrieve(url, dest_file_path)
urllib.request.urlretrieve(url, dest_file_path, reporthook=lambda count, blockSize, totalSize: print(f"Downloaded {count * blockSize / (1024 * 1024):.2f}/{totalSize / (1024 * 1024):.2f} MB ({100.0 * count * blockSize / totalSize:.1f}%)", end="\r"))

# extract the file
print("Extracting model weights...")
with zipfile.ZipFile(dest_file_path, 'r') as zip_ref:
    zip_ref.extractall()

# remove the downloaded zip file
os.remove(dest_file_path)

print("Model weights downloaded and extracted successfully!")