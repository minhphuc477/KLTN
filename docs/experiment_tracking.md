# Experiment Tracking

`src.utils.checkpoint.MetricsLogger` always writes the local JSON metrics log.
Weights & Biases forwarding is optional and does not replace that local record.

Enable forwarding with environment variables before launching a training run:

```powershell
$env:HMOLQD_WANDB_ENABLED = "1"
$env:HMOLQD_WANDB_PROJECT = "kltn-hmolqd"
# Optional:
$env:HMOLQD_WANDB_ENTITY = "your-entity"
$env:HMOLQD_WANDB_GROUP = "ablation-name"
```

If W&B is unavailable or initialization fails, training continues with the
local JSON log and emits a warning.
