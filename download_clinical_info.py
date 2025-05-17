import os
import requests

# Configuration
SAVE_PATH = "./data/raw/clinical_info.csv"
CLINICAL_INFO_URL = "https://api.vitaldb.net/cases"

# Create directory if needed
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Download and save
print(f"⬇️ Downloading clinical info from {CLINICAL_INFO_URL}...")
response = requests.get(CLINICAL_INFO_URL)

if response.status_code == 200:
    with open(SAVE_PATH, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved clinical info to {SAVE_PATH}")
else:
    print(f"❌ Failed to download. Status code: {response.status_code}")
