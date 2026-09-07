from __future__ import annotations

import numpy as np
import pandas as pd


def _tcn_model(torch, feature_count: int, hidden_channels: int):
    """Build a compact causal convolutional next-step forecaster."""
    nn = torch.nn

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, dilation):
            super().__init__()
            self.padding = 2 * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, 3,
                                  padding=self.padding, dilation=dilation)

        def forward(self, x):
            return self.conv(x)[:, :, :-self.padding]

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, dilation):
            super().__init__()
            self.layers = nn.Sequential(CausalConv1d(in_channels, out_channels, dilation), nn.ReLU(),
                                        CausalConv1d(out_channels, out_channels, dilation), nn.ReLU())
            self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            return torch.relu(self.layers(x) + self.skip(x))

    class TCNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(ResidualBlock(feature_count, hidden_channels, 1),
                                         ResidualBlock(hidden_channels, hidden_channels, 2),
                                         ResidualBlock(hidden_channels, hidden_channels, 4))
            self.head = nn.Linear(hidden_channels, feature_count)

        def forward(self, history):
            return self.head(self.network(history)[:, :, -1])

    return TCNForecaster()


def robust_multivariate_score(
    frame: pd.DataFrame, feature_columns: list[str], baseline_end: int
) -> pd.Series:
    """Return a label-free robust distance, fitted only on an initial baseline."""
    if baseline_end < 10:
        raise ValueError("baseline must contain at least 10 rows")
    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.iloc[:baseline_end].median().fillna(0))
    baseline = values.iloc[:baseline_end]
    median = baseline.median()
    mad = (baseline - median).abs().median().replace(0, np.nan)
    # 1.4826 makes MAD consistent with standard deviation under a Normal model.
    scale = (1.4826 * mad).fillna(baseline.std(ddof=0)).replace(0, 1.0).fillna(1.0)
    z = (values - median) / scale
    return np.sqrt((z**2).sum(axis=1)).rename("score")


def isolation_forest_score(frame, feature_columns, baseline_end, model_path):
    """Fit only normal baseline rows; persist preprocessing and fitted trees."""
    import joblib
    from sklearn.ensemble import IsolationForest

    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[:baseline_end].median().fillna(0)
    values = values.fillna(medians).to_numpy(dtype=np.float32)
    model = IsolationForest(n_estimators=200, max_samples=min(1024, baseline_end),
                            random_state=42, n_jobs=-1)
    model.fit(values[:baseline_end])
    joblib.dump({"model": model, "features": feature_columns, "medians": medians}, model_path)
    scores = np.concatenate([-model.score_samples(values[i:i + 50000])
                             for i in range(0, len(values), 50000)])
    return pd.Series(scores, index=frame.index, name="score")


def temporal_residual_score(frame, feature_columns, baseline_end, model_path):
    """Score one-step multivariate prediction residuals using only normal history."""
    import joblib
    from sklearn.linear_model import Ridge

    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[:baseline_end].median().fillna(0)
    values = values.fillna(medians)
    means = values.iloc[:baseline_end].mean()
    scales = values.iloc[:baseline_end].std(ddof=0).replace(0, 1).fillna(1)
    standardized = ((values - means) / scales).to_numpy(dtype=np.float32)
    model = Ridge(alpha=1.0)
    model.fit(standardized[:baseline_end - 1], standardized[1:baseline_end])
    predicted = model.predict(standardized[:-1])
    residuals = standardized[1:] - predicted
    baseline_residuals = residuals[:baseline_end - 1]
    residual_center = np.median(baseline_residuals, axis=0)
    residual_scale = 1.4826 * np.median(np.abs(baseline_residuals - residual_center), axis=0)
    residual_scale = np.where(residual_scale > 0, residual_scale, np.std(baseline_residuals, axis=0))
    residual_scale = np.where(residual_scale > 0, residual_scale, 1.0)
    scores = np.zeros(len(frame), dtype=np.float64)
    scores[1:] = np.sqrt(np.mean(((residuals - residual_center) / residual_scale) ** 2, axis=1))
    joblib.dump({"model": model, "features": feature_columns, "medians": medians,
                 "means": means, "scales": scales, "residual_center": residual_center,
                 "residual_scale": residual_scale}, model_path)
    return pd.Series(scores, index=frame.index, name="score")


def temporal_tcn_score(frame, feature_columns, baseline_end, model_path,
                       window_size=60, epochs=12, batch_size=512):
    """Causally predict the next sensor vector with a normal-only TCN fit."""
    try:
        import joblib
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise ImportError("temporal_tcn requires PyTorch; install the 'temporal' extra") from error
    if baseline_end <= window_size + 10:
        raise ValueError("baseline must contain at least window_size + 11 rows for temporal_tcn")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[:baseline_end].median().fillna(0)
    values = values.fillna(medians)
    means = values.iloc[:baseline_end].mean()
    scales = values.iloc[:baseline_end].std(ddof=0).replace(0, 1).fillna(1)
    standardized = ((values - means) / scales).to_numpy(dtype=np.float32)

    class WindowDataset(Dataset):
        def __init__(self, targets):
            self.targets = targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            target = self.targets[index]
            return standardized[target - window_size:target], standardized[target]

    # Fixed stride makes the normal-only fit feasible on million-row telemetry.
    targets = np.arange(window_size, baseline_end, 5, dtype=np.int64)
    loader = DataLoader(WindowDataset(targets), batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    model = _tcn_model(torch, len(feature_columns), hidden_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for history, target in loader:
            history, target = history.to(device), target.to(device)
            loss = torch.nn.functional.mse_loss(model(history.transpose(1, 2)), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    predictions = np.zeros_like(standardized, dtype=np.float32)
    with torch.no_grad():
        for start in range(window_size, len(standardized), batch_size):
            indices = np.arange(start, min(start + batch_size, len(standardized)))
            histories = np.stack([standardized[t - window_size:t] for t in indices])
            predictions[indices] = model(torch.from_numpy(histories).to(device).transpose(1, 2)).cpu().numpy()
    residuals = standardized[window_size:] - predictions[window_size:]
    baseline_residuals = residuals[:baseline_end - window_size]
    residual_center = np.median(baseline_residuals, axis=0)
    residual_scale = 1.4826 * np.median(np.abs(baseline_residuals - residual_center), axis=0)
    residual_scale = np.where(residual_scale > 0, residual_scale, np.std(baseline_residuals, axis=0))
    residual_scale = np.where(residual_scale > 0, residual_scale, 1.0)
    scores = np.zeros(len(frame), dtype=np.float64)
    scores[window_size:] = np.sqrt(np.mean(
        ((residuals - residual_center) / residual_scale) ** 2, axis=1))
    joblib.dump({"model_type": "causal_tcn_forecaster", "features": feature_columns,
                 "window_size": window_size, "hidden_channels": 64, "epochs": epochs,
                 "medians": medians, "means": means, "scales": scales,
                 "residual_center": residual_center, "residual_scale": residual_scale,
                 "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                model_path)
    return pd.Series(scores, index=frame.index, name="score")
