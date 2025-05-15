import os
import vitaldb

SAVE_DIR = './data/raw/vital_files'
os.makedirs(SAVE_DIR, exist_ok=True)

nsub = 6388

for caseid in range(1, nsub+1):
    filename = os.path.join(SAVE_DIR, f'{caseid}.vital')
    if os.path.exists(filename):
        print(f'✔️  Case {caseid} already downloaded, skipping.')
        continue

    try:
        print(f'⬇️  Downloading case {caseid}...')
        vf = vitaldb.VitalFile(caseid, ['SNUADC/ECG_II', 'SNUADC/PLETH'])
        vf.to_vital(filename)
        print(f'✅ Saved to {filename}')
    except Exception as e:
        print(f'❌ Error downloading case {caseid}: {e}')
