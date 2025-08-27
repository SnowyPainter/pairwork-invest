import kagglehub
import zipfile
import shutil
import os

# Create raw folder if it doesn't exist
raw_folder = "data/raw"
os.makedirs(raw_folder, exist_ok=True)

# Download and process Korean stock market data
korean_path = kagglehub.dataset_download("jwkhlee333/korean-stock-market-daily-data")
print(f"Korean data downloaded to: {korean_path}")

# Check if it's a zip file and extract if needed
if os.path.isfile(korean_path) and korean_path.endswith('.zip'):
    with zipfile.ZipFile(korean_path, 'r') as zip_ref:
        zip_ref.extractall(raw_folder)
    print(f"Korean data extracted to: {raw_folder}")
else:
    # Move the entire directory to raw folder
    korean_dest = os.path.join(raw_folder, "korean-stock-data")
    if os.path.exists(korean_dest):
        shutil.rmtree(korean_dest)
    shutil.move(korean_path, korean_dest)
    print(f"Korean data moved to: {korean_dest}")

# Download and process US stock market data
us_path = kagglehub.dataset_download("paultimothymooney/stock-market-data")
print(f"US data downloaded to: {us_path}")

# Check if it's a zip file and extract if needed
if os.path.isfile(us_path) and us_path.endswith('.zip'):
    with zipfile.ZipFile(us_path, 'r') as zip_ref:
        zip_ref.extractall(raw_folder)
    print(f"US data extracted to: {raw_folder}")
else:
    # Move the entire directory to raw folder
    us_dest = os.path.join(raw_folder, "us-stock-data")
    if os.path.exists(us_dest):
        shutil.rmtree(us_dest)
    shutil.move(us_path, us_dest)
    print(f"US data moved to: {us_dest}")
