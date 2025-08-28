import polars as pl

def future_return_labels(
    lf: pl.LazyFrame,
    horizon: int = 5,
    task: str = "regression",   # "regression" | "classification"
    thresh: float = 0.05
) -> pl.LazyFrame:
    g = "ticker"
    fut = pl.col("close").shift(-horizon).over(g)
    futret = (fut / pl.col("close") - 1).alias(f"futret_{horizon}")

    lf = lf.with_columns(futret)

    if task == "regression":
        return lf

    if task == "classification":
        label = (
            pl.when(pl.col(f"futret_{horizon}") >= thresh).then(1)
             .when(pl.col(f"futret_{horizon}") <= -thresh).then(-1)
             .otherwise(0)
             .cast(pl.Int8)
             .alias(f"label_{horizon}d_cls")
        )
        return lf.with_columns(label)

    raise ValueError("task must be regression or classification")
