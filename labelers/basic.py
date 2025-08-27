# labelers/basic.py
import polars as pl

def future_return_labels(
    lf: pl.LazyFrame,
    horizon: int = 5,
    task: str = "regression",   # "regression" | "classification"
    thresh: float = 0.02        # 분류시 임계 (예: ±2%)
) -> pl.LazyFrame:
    g = ["ticker"]
    lf = lf.with_columns([
        (pl.col("close").shift(-horizon)/pl.col("close") - 1).over(g).alias(f"futret_{horizon}")
    ])
    if task == "regression":
        return lf
    elif task == "classification":
        return lf.with_columns([
            (pl.when(pl.col(f"futret_{horizon}") >= thresh).then(1)
             .when(pl.col(f"futret_{horizon}") <= -thresh).then(-1)
             .otherwise(0)).alias(f"label_{horizon}d_cls")
        ])
    else:
        raise ValueError("task must be regression or classification")
