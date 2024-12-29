# Erkennung und Analyse von Shadowsocks-Traffic
# Detection and Analysis of Shadowsocks Traffic

## Autor / Author
- **Name:** Gino Peduto
- **Institution:** Universität Heidelberg / Heidelberg University 
- **Dokument / Document:** Bachelorarbeit / Bachelor Thesis
- **Jahr / Year:** 2025

## Projektbeschreibung / Project Description
[DE] Dieses Tool implementiert verschiedene Machine Learning Algorithmen zur Erkennung von Shadowsocks-Proxy-Traffic. Es nutzt NFStream für die Feature-Extraktion aus Netzwerk-Flows und implementiert mehrere Klassifikationsmodelle wie Random Forest, XGBoost, und andere zur Traffic-Klassifizierung.

[EN] This tool implements various machine learning algorithms to detect Shadowsocks proxy traffic. It uses NFStream for feature extraction from network flows and implements several classification models like Random Forest, XGBoost, and others for traffic classification.

### Hauptfunktionen / Key Features
- Feature-Extraktion aus PCAP-Dateien mittels NFStream / Feature extraction from PCAP files using NFStream
- Implementierung verschiedener ML-Klassifikatoren / Implementation of various ML classifiers
- Automatische Modellvergleiche und -evaluierung / Automatic model comparison and evaluation
- Traffic-Capture-Funktionalität / Traffic capture functionality
- Umfangreiche Visualisierung der Ergebnisse / Comprehensive result visualization
- SHAP-Analyse für Modellinterpretation / SHAP analysis for model interpretability
- Detaillierte Feature-Analyse und -Bewertung / Detailed feature analysis and evaluation

## Installation

### Requirements
- Python 3.8 or higher
- pip (Python Package Installer)
- NFStream and its dependencies

The project uses `pyproject.toml` for dependency management and build configuration.

### Basic Installation
For basic functionality:
```bash
pip install -e .
```

### Developer Installation
For developers with additional tools (tests, linting, etc.):
```bash
pip install -e .[dev]
```

### Full Installation
With all features including traffic capture:
```bash
pip install -e .[dev,capture]
```

## Project Structure
```
.
├── deployment/                # Deployment configurations
├── results/                   # Analysis outputs and model artifacts
│   ├── [experiment_name]/    # Individual experiment results
│   │   ├── features/         # Feature analysis outputs
│   │   │   ├── correlations/
│   │   │   ├── groups/
│   │   │   ├── importance/
│   │   │   └── summaries/
│   │   ├── models/           # Trained models
│   │   │   ├── LogisticRegression/
│   │   │   ├── RandomForestClassifier/
│   │   │   ├── SVC/
│   │   │   └── XGBClassifier/
│   │   ├── nfstream/         # NFStream results
│   │   │   ├── processed/
│   │   │   └── summaries/
│   │   ├── reports/          # Analysis reports
│   │   │   └── visualizations/
│   │   └── trained/          # Final trained models
│   │       └── best/
├── tcpdump_recorder/         # Modules for live traffic capture
│   ├── __init__.py
│   ├── no_proxy_firefox_recorder.py
│   ├── proton_recorder.py
│   ├── proxy_port_8388_recorder.py
│   └── tcpdump_recorder_all_trafic.py
├── tests/                    # Test modules
│   ├── __pycache__/
│   └── test_data/
├── traffic_analysis/         # Main analysis modules
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── data_inspector.py
│   ├── data_visualizer.py
│   ├── feature_analyzer.py
│   ├── main.py
│   ├── model_selection.py
│   ├── nfstream_feature_extractor.py
│   ├── output_manager.py
│   ├── shap_analyzer.py
│   └── sklearn_classifier.py
├── traffic_data/             # Raw data and intermediate files
│   ├── normal
│   ├── shadowsocks
│   ├── proton
│   ├── test
│   └── traffic_zips/
├── traffic_generation/       # Scripts for generating synthetic traffic
│   ├── configure_proxy.py
│   ├── surf_without_proxy.py
│   └── surf_with_ss_proxy.py
├── pyproject.toml            # Project configuration
└── README.md                 # Documentation (this file)

```

## Usage

### 1. Create PCABs - tcpdump_recorder
This script allows to quickly record pcaps, listening on port 443 for normal network traffic or port 8388 for network traffic over shadowsocks.
Optionally traffic_generation can be used - a small selenium script that simulates a user who wants to read messages on various relevant sites and browses around.

### 2. Run the machine learning pipeline - main.py
In main.py, an entry point named main() is defined near the bottom of the file, where three directory variables (proxy_dir, normal_dir, results_dir) can be customized.

```
proxy_dir = "/path/to/your/proxy_folder"

normal_dir = "/path/to/your/normal_folder"

results_dir = "/path/to/your/results_folder"

```
These directory paths should be set to the appropriate locations on the local system: one containing the proxy PCAP files, one containing the normal PCAP files, and a chosen folder for storing all analysis results.

- Execute the script, by running the main.py

now a pipeline with the following functions will start automatically:

  1) Copy PCAPs from original folders -> 'clean_data' subfolders
  2) Plot PCAP size distribution
  3) NFStream feature extraction
  4) Data cleaning
  5) Model training (GridSearch if param_grid != {})
  6) SHAP analysis if predict_proba is supported
  7) Compare model accuracies
  8) Save only the best model pipeline in 'trained/best'

a time-stamped subfolder will be created under the specified results_dir
there is also the best model for that run stored in a subfolder named **trained**, wich is a .joblib file

### 3. Make predictions
In the folder deployment is the file inference.py
At the beginning of the file inside the main-function are the necessary customisations to do
(PCAP folder and trained model path)

```
    # Adjust these paths as needed:
    pcap_dir = "/path/to/unlabeled_pcaps"    # <-- Place your PCAP directory path here
    model_path = "/path/to/BEST_XGBClassifier_pipeline.joblib"  # <-- Path to your joblib model
```

### Code Quality
```bash
# Format code
black .

# Type checking
mypy .

# Linting
flake8
```

### Configuration
Project configuration and dependencies are defined in `pyproject.toml`:
- Build System: setuptools
- Python Version: >=3.8
- Test Framework: pytest
- Code Formatting: black
- Type Checking: mypy

## Wissenschaftlicher Kontext / Academic Context
[DE] Diese Implementierung ist Teil meiner Bachelorarbeit "Erkennung und Analyse von Shadowsocks-Traffic" an der Universität Heidelberg. Die Arbeit untersucht Methoden zur Erkennung von verschlüsseltem Proxy-Traffic mittels maschinellen Lernens.

[EN] This implementation is part of my bachelor thesis "Detection and Analysis of Shadowsocks Traffic" at Heidelberg University. The work investigates methods for detecting encrypted proxy traffic using machine learning.

## Lizenz und Zitierung / License and Citation

### License
MIT License

Copyright (c) 2025 Gino Peduto

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Akademische Zitierung / Academic Citation
[DE] Bei Verwendung dieses Codes in akademischen Arbeiten bitte wie folgt zitieren:
[EN] When using this code in academic work, please cite as follows:

```bibtex
@misc{peduto2025shadowsocks,
  author = {Peduto, Gino},
  title = {Detection and Analysis of Shadowsocks Traffic},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Gino-Copilot/ba}},
  type = {Bachelor's Thesis},
  institution = {Heidelberg University}
}
```

Or in text form:
```
Peduto, G. (2025). Detection and Analysis of Shadowsocks Traffic [Bachelor's Thesis]. 
Heidelberg University. https://github.com/Gino-Copilot/ba
```
