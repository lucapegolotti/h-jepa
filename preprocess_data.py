import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import vitaldb
from tqdm import tqdm

# ---- Config ----
VITAL_PATH = "./data/raw/vital_files"
OUTPUT_FDR = "./data"
os.makedirs(OUTPUT_FDR, exist_ok=True)

OUTPUT_PATH_DATA = os.path.join(OUTPUT_FDR, "preprocessed_data.pt")
OUTPUT_PATH_IDS = os.path.join(OUTPUT_FDR, "preprocessed_ids.pt")

SIGNAL_KEYS = ["SNUADC/ECG_II", "SNUADC/PLETH"]
SRATE = 20  # target sample rate (Hz)
SAMPLE_DURATION_SECONDS = 30
WINDOW_SIZE = SAMPLE_DURATION_SECONDS * SRATE
STEP_SIZE = WINDOW_SIZE // 1  # overlap


def preprocess_vital_files():
    samples = []
    ids = []

    files = [
        os.path.join(VITAL_PATH, f)
        for f in os.listdir(VITAL_PATH)
        if f.endswith(".vital")
    ]
    for file in tqdm(files):
        try:
            vf = vitaldb.VitalFile(file, SIGNAL_KEYS)

            signals = []
            for sig in SIGNAL_KEYS:
                srate = vf.trks[sig].srate
                factor = max(1, round(srate / SRATE))
                signal = vf.to_numpy(sig, 0)[::factor]
                signals.append(signal)

            if any(len(s) < WINDOW_SIZE for s in signals):
                continue

            signals = [
                StandardScaler().fit_transform(s.reshape(-1, 1)).flatten()
                for s in signals
            ]
            min_len = min(map(len, signals))

            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                segment = np.stack(
                    [s[i : i + WINDOW_SIZE] for s in signals], axis=0
                )  # (C, T)
                if not np.isnan(segment).any():
                    samples.append(segment)
                    ids.append(int(os.path.basename(file).removesuffix(".vital")))

        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    print(f"✅ Extracted {len(samples)} samples")
    data = torch.tensor(np.stack(samples), dtype=torch.float32)
    ids = torch.tensor(ids, dtype=torch.int32)

    torch.save(data, OUTPUT_PATH_DATA)
    torch.save(ids, OUTPUT_PATH_IDS)
    print(f"📁 Saved data to {OUTPUT_PATH_DATA}")
    print(f"📁 Saved caseids to {OUTPUT_PATH_IDS}")


if __name__ == "__main__":
    preprocess_vital_files()
