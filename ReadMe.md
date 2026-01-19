# Anomaly Based Trading Strategies in Times of Crisis and Beyond

**Author:** Sanjana Kamwal <br/>
**Institution** University Of Freiburg <br/>
**Supervisor** Prof. Dr. Marcus Bravidor <br/>
**Thesis Title:** Anomaly Based Trading Strategies in Times of Crisis and Beyond<br/>
**Date:** January 2026<br/>
**Language:** Python<br/>

---

## Overview

This repository contains the codebase for the thesis _Anomaly Based Trading Strategies in Times of Crisis and Beyond_. The project performs a comprehensive financial analysis, including descriptive statistics, Welch's t-tests, and regression modeling (incorporating Fama-French factors and VIX interaction terms) to evaluate market anomalies.

The workflow is designed to be sequential, where data processed in the initial scripts is required for subsequent regression analyses.

---

## Installation & Setup

Please follow the instructions below to set up the development environment.

### 1. Environment Preparation

Ensure your terminal's working directory (`pwd`) is set to the root directory of this project.

### 2. Create Virtual Environment

Create a virtual environment to manage project dependencies in isolation.

- **macOS / Linux:**

```bash
python3 -m venv .venv

```

- **Windows:**

```bash
python -m venv .venv

```

### 3. Activate Virtual Environment

Activate the environment to ensure all libraries are installed locally.

- **Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat

```

- **Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1

```

- **macOS / Linux:**

```bash
source .venv/bin/activate

```

### 4. Install Dependencies

Once the environment is active, install the required packages using the provided requirements file:

```bash
pip install -r requirements.txt

```

---

## Usage Instructions

**Critical Note:** The files must be executed in the specific order outlined below. `descriptive_stats.py` processes raw data and exports intermediate files (e.g., `excess_returns.xlsx`) that are dependencies for the Jupyter Notebooks.

### Step 1: Descriptive Statistics

Run this script first to generate statistical tables and prepare regression data.

**How to Run in VS Code:**

1. Open `descriptive_stats.py`.
2. Right-click anywhere in the editor window.
3. Select **Run Python > Run Python File in Terminal**.

> **Disclaimer:** Please use the "Run Python File in Terminal" option. Using the generic "Run Code" button (often associated with the Code Runner extension) may ignore the `.venv` and attempt to use your global Python interpreter, resulting in errors.

**Generated Outputs:**

- **Folder:** `Descriptive_Stats_Results/`
- `descriptive_statistics_table.xlsx`
- `rolling_mean_excess_returns.png`
- `welchs_t_test_results.xlsx`

- **Folder:** `Regression_Data/`
- `excess_returns.xlsx`
- `fama_french_factors.xlsx`

### Step 2: Interaction Term Regression

1. Open `regression_with_interaction_term.ipynb`.
2. Ensure the notebook kernel is set to **.venv**.
3. Click **Run All**.

**Generated Outputs:**

- **Folder:** `Regression Results/`
- `model_1_revised_results.xlsx`
- `model_2_revised_results.xlsx`

### Step 3: VIX Regression

1. Open `vix_regression.ipynb`.
2. Ensure the notebook kernel is set to **.venv**.
3. Click **Run All**.

**Generated Outputs:**

- **Folder:** `VIX Regression Results/`
- `vix_model_results.xlsx`

---
