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
├── deployment/                # Deployment-Konfigurationen
├── results/                   # Analyse-Ergebnisse und Modell-Outputs
│   ├── [experiment_name]/    # Ergebnisse einzelner Experimente
│   │   ├── features/         # Feature-Analysen
│   │   │   ├── correlations/
│   │   │   ├── groups/
│   │   │   ├── importance/
│   │   │   └── summaries/
│   │   ├── models/          # Trainierte Modelle
│   │   │   ├── LogisticRegression/
│   │   │   ├── RandomForestClassifier/
│   │   │   ├── SVC/
│   │   │   └── XGBClassifier/
│   │   ├── nfstream/        # NFStream Ergebnisse
│   │   │   ├── processed/
│   │   │   └── summaries/
│   │   ├── reports/         # Analyseberichte
│   │   │   └── visualizations/
│   │   └── trained/         # Trainierte Modelle
│   │       └── best/
├── tcpdump_recorder/         # Traffic-Aufzeichnungsmodule
│   ├── __init__.py
│   ├── no_proxy_firefox_recorder.py
│   ├── proton_recorder.py
│   ├── proxy_port_8388_recorder.py
│   └── tcpdump_recorder_all_trafic.py
├── tests/                    # Testmodule
│   ├── __pycache__/
│   └── test_data/
├── traffic_analysis/         # Hauptanalysemodule
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
├── traffic_data/            # Rohdaten und verarbeitete Daten
│   ├── firefox_without_proxy_2024-12-08/
│   ├── normal_traffic_comparison_12-23/
│   ├── PROTON_test/
│   ├── regular_selenium_traffic/
│   ├── shadowsocks_traffic/
│   └── traffic_zips/
├── traffic_generation/      # Traffic-Generierungstools
│   ├── configure_proxy.py
│   ├── surf_without_proxy.py
│   └── surf_with_ss_proxy.py
├── pyproject.toml          # Projektkonfiguration
└── README.md               # Diese Datei

```

## Usage

### Feature Extraction and Model Training
```python
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS

# Feature extraction
extractor = NFStreamFeatureExtractor()
df = extractor.prepare_dataset(proxy_dir, normal_dir)

# Model training and evaluation
for model_name, model in MODELS.items():
    classifier = ScikitLearnTrafficClassifier(model)
    classifier.train(df)
```

### Results
Analysis results are saved in the following directories:
- `analysis_results/`: Visualizations and model comparisons
- `nfstream_results/`: Feature importance and datasets

## Development

### Running Tests
```bash
pytest
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
