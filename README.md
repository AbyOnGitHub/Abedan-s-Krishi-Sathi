# Farmer Recommendation Engine

This folder contains an AI Eligibility Scoring Engine built to evaluate farmers against various agricultural schemes and recommend the best options by computing a relevance and priority score.

## Directory Structure

* **`data/`**: Contains the input dataset `farmer_schemes.csv` containing up to 100 agricultural schemes.
* **`src/`**: Contains the python script `model.py` which implements the AI Eligibility Scoring Engine.
* **`scoring_methods.md`**: Provides a detailed breakdown of the mathematical formulas and heuristics utilized for scoring recommendations.

## How to Run

1. Ensure you have Python installed with the `pandas` library.
   ```bash
   pip install pandas
   ```
2. Navigate to the `src` folder and execute the model script:
   ```bash
   python model.py
   ```
3. The script will initialize the engine, load the data, and print out two simulated test cases outputting top scheme recommendations for dummy farmer profiles.
