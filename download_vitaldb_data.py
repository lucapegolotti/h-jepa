import os
import vitaldb
import requests
import yaml


def download_clinical_info(config):
    """
    Downloads clinical information from the VitalDB API and saves it to a specified file.

    Args:
        config (dict): A configuration dictionary containing the following keys:
            - "paths" (dict): A dictionary with the key "clinical_info_file" specifying
              the file path where the clinical information will be saved.

    Behavior:
        - Creates the necessary directories for the specified file path if they do not exist.
        - Sends a GET request to the VitalDB API to download clinical information.
        - Saves the downloaded data to the specified file path if the request is successful.
        - Prints status messages indicating the progress and outcome of the operation.

    Raises:
        - Prints an error message if the HTTP request fails, including the status code.
    """
    # Configuration
    CLINICAL_INFO_URL = "https://api.vitaldb.net/cases"

    # Create directory if needed
    os.makedirs(os.path.dirname(config["paths"]["clinical_info_file"]), exist_ok=True)

    # Download and save
    print(f"⬇️ Downloading clinical info from {CLINICAL_INFO_URL}...")
    response = requests.get(CLINICAL_INFO_URL)

    save_path = config["paths"]["clinical_info_file"]
    if response.status_code == 200:
        with open(config["paths"]["clinical_info_file"], "wb") as f:
            f.write(response.content)
        print(f"✅ Saved clinical info to {save_path}")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")


def download_vitaldb_waveforms(config):
    """
    Downloads waveform data from VitalDB for a specified range of case IDs.

    This function iterates through a predefined number of case IDs, checks if the
    corresponding waveform data file already exists in the specified directory,
    and downloads the data if it is not already present. The downloaded data is
    saved in the `.vital` format.

    Args:
        config (dict): A configuration dictionary containing the following keys:
            - "paths": A dictionary with the key "raw_data_dir" specifying the
              directory where the downloaded data will be stored.

    Behavior:
        - Creates the target directory if it does not already exist.
        - Skips downloading for case IDs that already have corresponding files
          in the target directory.
        - Downloads waveform data for the specified case IDs using the
          `vitaldb.VitalFile` class, focusing on the "SNUADC/ECG_II" and
          "SNUADC/PLETH" channels.
        - Handles and logs any exceptions that occur during the download process.
    """
    # Create directory if it doesn't exist
    os.makedirs(config["paths"]["raw_data_dir"], exist_ok=True)

    # Number of subjects to download
    nsub = 6388

    for caseid in range(1, nsub + 1):
        filename = os.path.join(config["paths"]["raw_data_dir"], f"{caseid}.vital")
        if os.path.exists(filename):
            print(f"✔️  Case {caseid} already downloaded, skipping.")
            continue

        try:
            print(f"⬇️  Downloading case {caseid}...")
            vf = vitaldb.VitalFile(
                caseid,
                [
                    "SNUADC/ECG_II",
                    "SNUADC/PLETH",
                    "CardioQ/CO",
                    "CardioQ/SV",
                    "SNUADC/ART",
                ],
            )
            vf.to_vital(filename)
            print(f"✅ Saved to {filename}")
        except Exception as e:
            print(f"❌ Error downloading case {caseid}: {e}")


if __name__ == "__main__":
    CONFIG_PATH = "config.yml"

    print("📖 Loading configuration...")
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    download_clinical_info(config)
    download_vitaldb_waveforms(config)
