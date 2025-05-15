### 📁 `README.md`

# VitalDB-JEPA

A self-supervised representation learning pipeline using JEPA (Joint Embedding Predictive Architecture) on physiological signals from the [VitalDB dataset](https://vitaldb.net). This project trains an encoder to predict masked regions of ECG and PPG signals, then evaluates the learned embeddings by regressing clinical outcomes (e.g., age).

## 🧪 Setup

```bash
bash create_venv.sh
source vitaldb_jepa_env/bin/activate
````

Or install manually:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm plotly vitaldb
```

## 🚀 Workflow

### 1. Download VitalDB Data

```bash
python download_clinical_info.py      # Saves clinical_info.csv
python download_vitaldb.py            # Downloads .vital waveform files
```

### 2. Preprocess

```bash
python preprocess_data.py             # Outputs: preprocessed_data.pt, preprocessed_ids.pt
```

### 3. Train JEPA

```bash
python train.py                       # Trains the encoder, saves jepa_model.pth
```

### 4. Evaluate Embeddings

```bash
python regression_probe.py            # Predicts age from embeddings + plots performance
```

---

## ⚙️ Training Details

* Input: (ECG + PPG) segments 
* Architecture: 1D ConvNet with adaptive pooling
* JEPA masking: random region of within each segment
* Loss: cosine distance between predicted and actual embeddings

---

## 📈 Evaluation

Embedding quality is assessed via downstream regression of:

* Age
* Weight
* Cardiac Output (planned)

Results are visualized in a 2D scatter plot.

---

## 📋 TODO

* [ ] Add support for classification tasks (e.g., ASA class, age bins)
* [ ] Support variable-length masking
* [ ] Experiment with other biosignals (e.g., ART, SPO2)
* [ ] Try nonlinear probe models (MLP, XGBoost)

---

## 📚 Citation

If you use this pipeline with VitalDB data, please cite:

> Lee HC, Jung CW. VitalDB: A high-fidelity multi-parameter vital signs database in surgical patients. Sci Data. 2020.