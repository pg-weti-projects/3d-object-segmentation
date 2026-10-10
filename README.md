# 3D Swim Bladder Segmentation & Reconstruction

A research framework for **3D swim bladder segmentation from CT scans**, geometric mesh reconstruction, and multi-model benchmarking.

---

## Flask Application

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The `src/api/config.toml` file contains application runtime settings. Adjust it
if needed, then start Flask:

```bash
python -m src.api.app
```

Open `http://127.0.0.1:5000` to verify that the standalone web application is
running.

## 📁 Project Structure

> **Note:** This is the planned target structure. Some components are not implemented yet.

```text
├── configs/
│   ├── default_config.yaml           # base configuration
│   └── experiments/                  # experiment-specific YAML configs e.g., exp_002_unet.yaml
├── data/
│   ├── raw/                          # input CT scans (.dcm, .tif/.tiff, .json)
│   └── reference_masks/              # ground truth masks from 3D Slicer (.nii.gz)
├── results/                          # generated experiment outputs
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── app.py                    # Flask app factory and entry point
│   │   ├── config.py                 # TOML configuration loading
│   │   ├── config.toml               # application runtime settings
│   │   ├── routes/
│   │   │   └── web.py                # HTML routes
│   │   ├── services/                 # application services
│   │   ├── templates/
│   │   │   ├── base.html             # shared page layout
│   │   │   └── index.html            # application entry page
│   │   └── static/
│   │       ├── css/
│   │       │   └── app.css            # application styles
│   │       └── js/
│   │           ├── app.js             # shared frontend behavior
│   │           └── webgl-viewer.js    # WebGL viewer
│   ├── core/                         # data loading & preprocessing
│   │   ├── volume.py                 # unified Volume (dataclass) representation
│   │   ├── loaders.py                # DICOM, TIFF & JSON loaders
│   │   └── preprocessor.py           # normalization, resampling & denoising
│   ├── models/                       # Segmentation models 
│   │   ├── analytical.py             
│   │   ├── unet.py                   
│   │   └── foundation/               
│   └── pipeline/                     # reconstruction, metrics & experiments
│       ├── mesh.py                   # Marching Cubes + smoothing + decimation
│       ├── metrics.py                # Dice, IoU, HD95, ASSD & VS
│       └── runner.py                 # deterministic experiment runner
├── tests/                            # unit & integration tests
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```
---
