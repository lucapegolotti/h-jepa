import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import vitaldb
from tqdm import tqdm
import yaml

# ---- Load Config ----
CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Ensure output directories exist
os.makedirs(os.path.dirname(config["paths"]["preprocessed_data"]), exist_ok=True)
os.makedirs(os.path.dirname(config["paths"]["preprocessed_ids"]), exist_ok=True)

def preprocess_vital_files():
    samples = []
    ids = []

    files = [
        os.path.join(config["paths"]["raw_data_dir"], f)
        for f in os.listdir(config["paths"]["raw_data_dir"])
        if f.endswith(".vital")
    ]
    for file in tqdm(files):
        try:
            vf = vitaldb.VitalFile(file, config["preprocessing"]["signal_keys"])

            signals = []
            for sig in config["preprocessing"]["signal_keys"]:
                srate = vf.trks[sig].srate
                factor = max(1, round(srate / config["preprocessing"]["sample_rate"]))
                signal = vf.to_numpy(sig, 0)[::factor]
                signals.append(signal)

            window_size = config["preprocessing"]["sample_duration_seconds"] * config["preprocessing"]["sample_rate"]
            step_size = config["preprocessing"].get("step_size", window_size // 1)

            if any(len(s) < window_size for s in signals):
                continue

            signals = [
                StandardScaler().fit_transform(s.reshape(-1, 1)).flatten()
                for s in signals
            ]
            min_len = min(map(len, signals))

            for i in range(0, min_len - window_size, step_size):
                segment = np.stack(
                    [s[i : i + window_size] for s in signals], axis=0
                )  # (C, T)
                if not np.isnan(segment).any():
                    samples.append(segment)
                    ids.append(int(os.path.basename(file).removesuffix(".vital")))

        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    print(f"✅ Extracted {len(samples)} samples")
    data = torch.tensor(np.stack(samples), dtype=torch.float32)
    ids = torch.tensor(ids, dtype=torch.int32)

    torch.save(data, config["paths"]["preprocessed_data"])
    torch.save(ids, config["paths"]["preprocessed_ids"])
    print(f'📁 Saved data to {config["paths"]["preprocessed_data"]}')
    print(f'📁 Saved caseids to {config["paths"]["preprocessed_ids"]}')

if __name__ == "__main__":
    preprocess_vital_files()
