# DeBiaFuse

Bridge deflection forecasting experiments for the four Hongfu monitoring
points: `DD-EN-09`, `DD-ES-09`, `DD-WN-09`, and `DD-WS-09`.

## Environment

- Python 3.11+
- PyTorch
- Dependencies: `pip install -r requirements.txt`

## Reproducible baseline comparison

The current comparison uses the data under
`DLA/data/Hongfu/deflection/`, daily aggregation, a chronological 70/10/20
train/validation/test split, and a Min-Max scaler fitted on the training split
only. Every model uses `look_back=24` and `horizon=6`; metrics are calculated
after inverse transformation in the original deflection units.

Run the five-model comparison (Persistence, ARIMA, LSTM, DLinear, and
Crossformer) with:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=code \
python code/run_baselines.py --epochs 50 \
  --models persistence,arima,lstm,dlinear,crossformer \
  --output results/baselines_hongfu.json
```

Results are written to [results/baselines_hongfu.json](results/baselines_hongfu.json)
and [results/baselines_hongfu.csv](results/baselines_hongfu.csv).

### Recorded results (50 epochs)

| Dataset | Persistence MAE | ARIMA MAE | LSTM MAE | DLinear MAE | Crossformer MAE |
|---|---:|---:|---:|---:|---:|
| DD-EN-09 | **2.048** | 10.877 | 2.744 | 2.627 | 2.431 |
| DD-ES-09 | **2.186** | 11.489 | 3.133 | 2.779 | 2.784 |
| DD-WN-09 | **1.756** | 9.873 | 2.536 | 2.192 | 2.127 |
| DD-WS-09 | **2.478** | 12.090 | 3.486 | 2.969 | 2.913 |

Under this leakage-safe protocol, Persistence is the strongest model on all
four series. Crossformer is the strongest learned model overall, followed by
DLinear; ARIMA is substantially worse on this short, non-stationary daily
dataset.

## Other entry points

- `code/DeBiaFuse.py`: legacy DeBiaFuse training entry point.
- `code/run_hongfu_p0.py`: leakage-safe LSTM and Persistence P0 experiment.
- `code/debiafuse_pipeline.py`: split, scaling, causal decomposition, and
  factorized biaxial-attention utilities.
- `code/tests/`: data-pipeline and model-shape tests.
