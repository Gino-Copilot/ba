# Erkennung und Analyse von Shadowsocks-Traffic

Dieses Repository enthält den Code für meine Bachelorarbeit an der Universität Heidelberg. Die Implementierung fokussiert sich auf die Erkennung und Analyse von Shadowsocks-Traffic mittels Machine Learning Methoden.

## Autor
- **Name:** Gino Peduto
- **Institution:** Universität Heidelberg
- **Dokument:** Bachelorarbeit
- **Jahr:** 2024

## Projektbeschreibung
Dieses Tool implementiert verschiedene Machine Learning Algorithmen zur Erkennung von Shadowsocks-Proxy-Traffic. Es nutzt NFStream für die Feature-Extraktion aus Netzwerk-Flows und implementiert mehrere Klassifikationsmodelle wie Random Forest, XGBoost, und andere zur Traffic-Klassifizierung.

### Hauptfunktionen
- Feature-Extraktion aus PCAP-Dateien mittels NFStream
- Implementierung verschiedener ML-Klassifikatoren
- Automatische Modellvergleiche und -evaluierung
- Traffic-Capture-Funktionalität
- Umfangreiche Visualisierung der Ergebnisse

## Installation

### Voraussetzungen
- Python 3.8 oder höher
- pip (Python Package Installer)
- NFStream und seine Abhängigkeiten

Das Projekt verwendet `pyproject.toml` für Dependency-Management und Build-Konfiguration.

### Basis-Installation
Für die grundlegende Funktionalität:
```bash
pip install -e .
```

### Entwickler-Installation
Für Entwickler mit zusätzlichen Tools (Tests, Linting, etc.):
```bash
pip install -e .[dev]
```

### Vollständige Installation
Mit allen Funktionen inkl. Traffic-Capture:
```bash
pip install -e .[dev,capture]
```

## Projektstruktur
```
.
├── traffic_analysis/          # Hauptmodul für die Verkehrsanalyse
│   ├── main.py               # Hauptausführungsdatei
│   ├── model_selection.py    # ML-Modellkonfigurationen
│   ├── sklearn_classifier.py # ML-Klassifikatoren
│   └── nfstream_feature_extractor.py  # Feature-Extraktion
├── traffic_capture/          # Module für Traffic-Aufzeichnung
│   └── tcpdump_recorder.py   # Traffic-Capture-Funktionalität
├── tests/                    # Testmodule
├── pyproject.toml           # Projekt-Konfiguration und Dependencies
└── README.md                # Diese Datei
```

## Verwendung

### Feature-Extraktion und Modelltraining
```python
from traffic_analysis.nfstream_feature_extractor import NFStreamFeatureExtractor
from traffic_analysis.sklearn_classifier import ScikitLearnTrafficClassifier
from traffic_analysis.model_selection import MODELS

# Feature-Extraktion
extractor = NFStreamFeatureExtractor()
df = extractor.prepare_dataset(proxy_dir, normal_dir)

# Modelltraining und Evaluation
for model_name, model in MODELS.items():
    classifier = ScikitLearnTrafficClassifier(model)
    classifier.train(df)
```

### Ergebnisse
Die Analyseergebnisse werden in folgenden Verzeichnissen gespeichert:
- `analysis_results/`: Visualisierungen und Modellvergleiche
- `nfstream_results/`: Feature-Importance und Datensätze

## Entwicklung

### Tests ausführen
```bash
pytest
```

### Code-Qualität
```bash
# Code formatieren
black .

# Type checking
mypy .

# Linting
flake8
```

### Konfiguration
Die Projekt-Konfiguration und Abhängigkeiten sind in `pyproject.toml` definiert:
- Build-System: setuptools
- Python-Version: >=3.8
- Test-Framework: pytest
- Code-Formatierung: black
- Type-Checking: mypy

## Wissenschaftlicher Kontext
Diese Implementierung ist Teil meiner Bachelorarbeit "Erkennung und Analyse von Shadowsocks-Traffic" an der Universität Heidelberg. Die Arbeit untersucht Methoden zur Erkennung von verschlüsseltem Proxy-Traffic mittels maschinellen Lernens.

## Lizenz und Zitierung

### Lizenz
MIT License

Copyright (c) 2024 Gino Peduto

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Akademische Zitierung
Bei Verwendung dieses Codes in akademischen Arbeiten bitte wie folgt zitieren:

```bibtex
@misc{peduto2024shadowsocks,
  author = {Peduto, Gino},
  title = {Erkennung und Analyse von Shadowsocks-Traffic},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Gino-Copilot/ba}},
  type = {Bachelor's Thesis},
  institution = {Universität Heidelberg}
}
```

Oder in textueller Form:
```
Peduto, G. (2024). Erkennung und Analyse von Shadowsocks-Traffic [Bachelor's Thesis]. 
Universität Heidelberg. https://github.com/Gino-Copilot/ba
```

---
© 2024 Gino Peduto - Universität Heidelberg