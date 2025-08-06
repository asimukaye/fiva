# FIVA: Federated Inverse Variance Aggretation
## Code Structure

```
fed_learning/fiva/
├── README.md
├── data/
│   └── ...                # Data files for federated learning
├── models.py              # Model definitions
├── utils.py               # Utility scripts
├── fiva.py                # FIVA aggregation algorithm
├── main.py                # Entry point for running experiments
├── configs/               # Configuration files
└── requirements.txt       # Python dependencies
```

## Instructions

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Prepare your data:**
    - Place your datasets in the `data/` directory.

3. **Run an experiment:**
    ```bash
    python main.py --config configs/your_config.yaml
    ```
    - The `main.py` script loads configuration, initializes models, runs federated training, and applies FIVA aggregation.

4. **Modify configuration:**
    - Edit or create YAML files in the `configs/` directory to customize experiments.

## Example

```bash
python main.py 
```

## Notes

- The FIVA aggregation algorithm is implemented in `fiva.py`.
- Model architectures are in `models.py`.
- Utility functions are in `utils.py`.
- For more details, see comments in each module.

## Reference

