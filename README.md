# DeBiaFuse

Bridge deflection forecasting experiments for the four Hongfu monitoring
points: `DD-EN-09`, `DD-ES-09`, `DD-WN-09`, and `DD-WS-09`.

## Environment

- Python 3.11+
- PyTorch
- Dependencies: `pip install -r requirements.txt`

## Reproducible baseline comparison

The current comparison uses the data under
`data/Hongfu/deflection/`, daily aggregation, a chronological 70/10/20
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

## DeBiaFuse v2 leakage-safe experiment

The v2 model uses a causal trend LSTM plus window-local EMD residual
components and joint temporal/component biaxial attention. It can be run in
direct or residual-to-persistence mode:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=code \
python code/train_debiafuse.py --epochs 50 --target direct \
  --output results/debiafuse_hongfu_seed42_direct.json
```

The recorded v2 results are available in:

- [direct JSON](results/debiafuse_hongfu_seed42_direct.json)
- [direct CSV](results/debiafuse_hongfu_seed42_direct.csv)
- [residual JSON](results/debiafuse_hongfu_seed42_residual.json)
- [residual CSV](results/debiafuse_hongfu_seed42_residual.csv)
- [data quality report](results/data_quality_hongfu.csv)

### Recorded DeBiaFuse v2 results (50 epochs, MAE)

| Dataset | Persistence | DeBiaFuse direct | DeBiaFuse residual |
|---|---:|---:|---:|
| DD-EN-09 | **2.048** | 2.768 | 3.636 |
| DD-ES-09 | **2.186** | 3.435 | 4.283 |
| DD-WN-09 | **1.756** | 2.395 | 2.775 |
| DD-WS-09 | **2.478** | 3.646 | 3.977 |

Under the strict leakage-safe protocol, DeBiaFuse v2 runs end to end, but
neither direct nor residual mode currently beats Persistence. The direct mode
is consistently better than residual mode. These results are retained as the
current correctness checkpoint; further architecture additions should wait
until the model demonstrates positive skill over Persistence.

## DeBiaFuse v3 correctness results

The v3 rerun adds a complete daily calendar with causal forward-fill,
forecast-origin-only component masks, coherent residual targets, separate
decomposition context and trend window, and horizon-wise metrics. Results are
stored separately from the v2 checkpoint:

- `results/debiafuse_v3_24to6_direct.json/csv`
- `results/debiafuse_v3_24to6_residual.json/csv`
- `results/debiafuse_v3_60to30_direct.json/csv`
- `results/debiafuse_v3_60to30_residual.json/csv`

| Setting | Direct average Skill_MAE | Residual average Skill_MAE |
|---|---:|---:|
| 24→6 | approximately -0.25 | approximately -0.03 |
| 60→30 | approximately -0.67 | approximately +0.03 |

The corrected residual formulation is substantially better than the v2
residual checkpoint. The strongest current result is 60→30 residual mode,
which achieves positive skill on DD-ES-09, DD-WN-09 and DD-WS-09, including
approximately +0.17 on DD-WS-09. The 24→6 settings still do not beat
Persistence, so these results remain a correctness checkpoint rather than a
final DSE benchmark claim.
