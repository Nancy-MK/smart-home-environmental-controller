# Smart Home Environmental Controller - AI Reliability & Failure-Mode Review

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![TimeSeries](https://img.shields.io/badge/task-time--series-green)

An adaptive environmental control system (think: smart thermostat / HVAC controller) evaluated the way a safety-relevant control system should be: not just "does the forecaster predict well", but **what happens when it's wrong, and does the system fail safely?**

## Problem framing

Forecasting models embedded in physical control loops (heating, cooling, ventilation) do not just need to be accurate on average, they need documented behaviour under drift, sensor noise, and conflicting control objectives. This project benchmarks four forecasting architectures for reliability and drift risk, then separately verifies that the control logic built on top of the chosen forecaster degrades safely rather than unpredictably.

## Forecasting benchmark

Four architectures were benchmarked on 15-minute-ahead environmental forecasting (e.g. indoor temperature):

| Model | Type |
|---|---|
| Linear Regression | Baseline, fully interpretable |
| RNN | Recurrent baseline |
| GRU | Gated recurrent unit |
| LSTM | Long short-term memory |

Each model is evaluated on **RMSE and MAE across 15-minute prediction intervals**, and the "best" model is selected using **risk-adjusted performance criteria**, not raw error alone: a slightly less accurate but more stable model (lower variance in error across conditions) may be preferable to a marginally more accurate but erratic one in a system that directly drives physical heating/cooling actuators.

## Failure-mode review

The adaptive control logic sitting on top of the forecaster was reviewed for **constraint-conflict failure modes** - situations where, for example, a comfort-temperature target and an energy-saving constraint cannot both be satisfied. The review verifies:

- The system has a defined **safe fallback behaviour** (e.g. reverting to a conservative, pre-set schedule) when constraints conflict or forecasts are flagged as unreliable
- This fallback triggers **without requiring human intervention**, addressing operational continuity risk in an unattended AI-driven system
- Fallback conditions and expected behaviour are documented, the same deliverable format used for human-oversight and failure-mode documentation in AI governance frameworks

## Repository structure

```
smart-home-environmental-controller/
  README.md
  requirements.txt
  src/
    data_prep.py         # windowing and train/test split for time-series data
    train_forecasters.py # trains and benchmarks Linear/RNN/GRU/LSTM models
    controller.py         # adaptive control logic with safe-fallback behaviour
    failure_mode_tests.py # constraint-conflict and fallback verification tests
```

## Getting started

```bash
git clone https://github.com/Nancy-MK/smart-home-environmental-controller.git
cd smart-home-environmental-controller
pip install -r requirements.txt

# 1. Prepare windowed time-series data
python src/data_prep.py --data data/sensor_readings.csv --window 12

# 2. Train and benchmark all four forecasters
python src/train_forecasters.py --data data/sensor_readings.csv

# 3. Run failure-mode / safe-fallback verification tests
python src/failure_mode_tests.py --model artifacts/best_model.keras
```

The input CSV is expected to contain a timestamp column and one or more sensor readings (e.g. `temperature`, `humidity`) sampled at regular intervals.

## Skills demonstrated

- Time-series forecasting (Linear Regression, RNN, GRU, LSTM)
- Risk-adjusted model selection (not accuracy alone)
- Failure-mode analysis (FMEA-style) for a control system
- Safe-fallback / human-oversight-free continuity design

## Tech stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy

## Licence

Developed for academic purposes. All rights reserved (c) Nancy Kamal.
