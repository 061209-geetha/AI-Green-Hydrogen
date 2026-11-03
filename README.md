# AI-Enhanced Integration of Green Hydrogen in Urban Infrastructure for Carbon-Neutral Cities

**GitHub Repository README | Python Research Implementation**

A reproducible, software-only research simulator integrating hydrogen-equivalent demand forecasting, GIS-based station and corridor planning, and carbon-emission scenario analysis. The package includes synthetic demo inputs, real-data adapters, benchmarks, automated figures, and unit tests.

## 1. Project Overview

The pipeline transforms half-hourly household electricity data and weather-dependent heating assumptions into daily hydrogen-equivalent demand. A stacked LSTM produces next-day forecasts, which are aggregated into planning zones. A capacity- and radius-constrained siting model chooses infrastructure sites, and graph-based routing estimates connected pipeline corridors. The carbon module evaluates emissions under partial and full hydrogen-substitution scenarios.

## 2. Included Features

- Stacked LSTM forecaster (128 and 64 hidden units, 24-day lookback) with both PyTorch and optional TensorFlow/Keras backends.
- Hydrogen conversion with configurable fuel-cell efficiency, lower heating value, degree-day heating and gas-demand calibration.
- Chronological training/validation/test splits, forecast metrics, model checkpoints, and comparison models.
- OSMnx/NetworkX real-area support, SciPy-HiGHS capacitated station siting, and shared-corridor approximation.
- Conventional fixed-four-station Euclidean p-median baseline with separate radial routes.
- Carbon inventory, substitution scenarios (25%, 50%, 75%, 100%), hydrogen intensity sensitivity, and coverage-aware reporting.
- Optional SHAP (for the PyTorch backend), gradient saliency figures, five-seed experiment launcher, 600-dpi plots, and unit tests.

## 3. Repository Layout

```text
AI_Green_Hydrogen/
├── README.md                  # Native GitHub Markdown README
├── HOWTORUN.txt               # Detailed execution instructions
├── requirements.txt           # Core dependencies
├── requirements-geo.txt       # Optional geospatial packages
├── requirements-research.txt  # Additional research dependencies
├── configs/default.yaml       # Experiment parameters
├── run_pipeline.py            # Main pipeline CLI
├── run_experiments.py         # Repeated-seed orchestration
├── hydrogen_city/             # Data, LSTM, GIS, carbon, evaluation modules
├── tests/                     # Automated unit tests
└── examples/                  # Included example artifacts
```

## 4. Environment and Installation

Python 3.11 is recommended. Install the core stack in a virtual environment:

```bash
python -m venv .venv

# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

python -m pip install -r requirements.txt
```

For real OpenStreetMap processing, install the optional GIS stack. For expanded research analyses, install the research requirements.

```bash
python -m pip install -r requirements-geo.txt
python -m pip install -r requirements-research.txt
```

The core dependencies include NumPy, pandas, SciPy, scikit-learn, NetworkX, Matplotlib, PyYAML, PyTorch, OpenPyXL, and Requests. TensorFlow is optional and must be installed separately for the Keras recurrent-dropout configuration.

## 5. Quick Start — Offline Synthetic Demonstration

This is the recommended first run because it requires no external dataset downloads. All generated inputs in this mode are synthetic and must not be presented as observed London results.

```bash
python run_pipeline.py --mode demo --epochs 3 --n-households 20 --days 110 --output outputs/demo_smoke

python -m unittest discover -s tests -v
```

Run a larger demonstration:

```bash
python run_pipeline.py --mode demo --epochs 30 --n-households 80 --days 380 --output outputs/demo_full
```

## 6. Real-Data Execution

Supply Low Carbon London electricity data, the UCI Energy Efficiency dataset, and complete daily weather for the meter-data time span. OpenStreetMap acquisition requires internet access. Bounding-box arguments are ordered NORTH SOUTH EAST WEST; the coordinates below are illustrative and must be replaced by an archived case-study boundary.

```bash
python run_pipeline.py --mode real --lcl /path/to/LCL.csv --uci /path/to/ENB2012_data.xlsx \
  --weather-csv /path/to/london_weather.csv \
  --bbox 51.57 51.525 -0.00 -0.08 \
  --max-real-households 250 --epochs 100 --output outputs/london
```

The daily weather CSV requires date and temperature_c columns (tavg is also accepted). Alternatively, install Meteostat and replace --weather-csv with --fetch-weather. The LCL dataset has electricity readings but no direct residential gas measurements or exact household geolocations.

## 7. Model Backends and Experiment Options

| Task | Command / configuration |
|---|---|
| **TensorFlow backend** | `pip install tensorflow; python run_pipeline.py --mode demo --backend tensorflow --epochs 20` |
| **Full comparator suite** | `python run_pipeline.py --mode demo --epochs 30 --full-benchmarks --benchmark-epochs 15 --output outputs/benchmarks` |
| **Five independent seeds** | `python run_experiments.py --seeds 0 1 2 3 4 --mode demo --epochs 100 --output outputs/five_seeds` |
| **SHAP explanations** | `pip install shap; python run_pipeline.py --mode demo --epochs 30 --shap` |
| **Skip benchmark fitting** | `python run_pipeline.py --mode demo --skip-benchmarks` |
| **Forecast-driven GIS** | `configs/default.yaml → spatial.demand_source: forecast (default)` |
| **Reference-inventory GIS** | `configs/default.yaml → spatial.demand_source: reference_inventory` |

The PyTorch model omits Keras-style recurrent dropout; the optional TensorFlow backend implements recurrent_dropout=0.1. Standard gradient attribution is not SHAP; SHAP is only identified as such when explicitly run.

## 8. Reproducibility and Core Configuration

Primary settings live in configs/default.yaml: lookback 24 days; Adam learning rate 0.001; batch size 32; early-stopping patience 10; base ηFC = 0.50; H₂ lower heating value 33.33 kWh/kg; heating degree-day base 15.5 °C; 10 zones, 38 candidate sites, service radius 3.5 km, maximum six stations, and capacity 12 t H₂/day per station. The nominal study-area inventory uses 25,400 dwellings. Emission factors default to 0.20705 kg CO₂e/kWh for electricity, 0.18290 for gas, and 0.75 kg CO₂e/kg H₂ for production.

Run manifests record selected parameters, software environment, data source mode, and modeling assumptions. Synthetic allocations, provisional geographic coordinates, and unverified parcel eligibility must be kept distinct from observed inputs.

## 9. Generated Outputs

```text
outputs/<run>/summary.json
outputs/<run>/run_manifest.json
outputs/<run>/spatial_results.json
outputs/<run>/tables/*.csv
outputs/<run>/figures/*.png
outputs/<run>/checkpoints/lstm_best.pt   # or .keras for TensorFlow
```

Outputs include forecasting predictions and metrics, scenario tables, GIS and emission plots, checkpoints, and runtime metadata. Results are calculated from each run’s inputs; the code does not hard-code the manuscript’s reported evaluation outcomes.

## 10. Known Limitations and Scientific Scope

- The exact 5 × 5 km georeferenced London polygon, validated 38 station plots, supply-hub point, and anonymised household-to-zone mapping were not supplied with the manuscript. Automatically sampled GIS sites must not be represented as validated buildable parcels.
- Gas demand is degree-day-modelled, not measured by the Low Carbon London electricity dataset. The ACORN-to-UCI building-archetype crosswalk is an explicit modelling assumption.
- Spatial site-selection optimality is based on an assignment-level approximation. The resulting shared-corridor network is evaluated separately and is not a proof of globally optimal integrated routing.
- The additional benchmark suite provides reference model implementations rather than exact reproduction of all manuscript comparators. Per-household SARIMA and every spatial comparator are not included.
- This repository is intended for methodological evaluation and research simulation; it is not a pipeline design, hydrogen safety assessment, or engineering certification tool.

## 11. Dataset References

- UCI Energy Efficiency: [https://archive.ics.uci.edu/dataset/242/energy+efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- Low Carbon London: UK Power Networks trial / London Datastore distribution (verify current host and access terms)
- OpenStreetMap: [https://www.openstreetmap.org](https://www.openstreetmap.org)
- IEA Hydrogen Projects Database: [https://www.iea.org/data-and-statistics/data-product/hydrogen-projects-database](https://www.iea.org/data-and-statistics/data-product/hydrogen-projects-database)
- UK conversion factors: [https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting](https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting)
- Meteostat: [https://meteostat.net](https://meteostat.net)

## 12. Citation, Licensing, and Repository Maintenance

When publishing results, cite the associated manuscript and original dataset sources. No open-source licence is declared by the supplied repository; add a LICENSE file only after the project owner selects a licence. Contributions, issue reporting instructions, author affiliations, DOI, and a publication citation can be added after those details are confirmed. Do not present synthetic smoke-test values as independently reproduced manuscript results.
