# Notebooks

This directory will contain Jupyter / Colab notebooks for interactive exploration:

- `01_quickstart.ipynb` — Quick introduction to using the SLATE package
- `02_reproduce_validation.ipynb` — End-to-end reproduction of the May 2026 validation results
- `03_entropy_test.ipynb` — Step-by-step walk-through of the information monotonicity test

## Run on Google Colab

The validation was originally performed on Google Colab Pro with an A100 GPU. To reproduce:

1. Open a new Colab notebook
2. Set runtime → GPU → A100 (or any available GPU; T4 will work but slowly)
3. Install dependencies:

   ```python
   !pip install native-sparse-attention-pytorch einops rotary-embedding-torch
   !git clone https://github.com/physivitis/slate.git
   %cd slate
   ```

4. Run the experiment scripts directly:

   ```python
   !python experiments/run_c3_strict.py
   !python experiments/run_entropy_test.py
   ```

Each script downloads Tiny Shakespeare automatically on first run.

## Notebook contributions

If you produce useful notebooks demonstrating SLATE on different datasets, model sizes, or applications, please contribute them via pull request.
