"""Public real-world datasets for the harness.

nyc_taxi: NYC TLC yellow-taxi trip records (public, no login). One month is ~3M rows / ~50 MB; 12 months is
~38M rows, which is where a GPU starts to matter. Months are concatenated into ONE parquet with a fixed schema.

  python -m evals.real.datasets nyc_taxi --months 1          # smoke test (~3M rows)
  python -m evals.real.datasets nyc_taxi --months 12         # scale test (~38M rows)
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
YEAR = 2023
DATA_DIR = Path("eval_data")


def fetch_nyc_taxi(months: int, dest: Path = DATA_DIR) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"nyc_taxi_{YEAR}_{months}m.parquet"
    if out.exists():
        return out
    writer, schema = None, None
    try:
        for m in range(1, months + 1):
            part = dest / f"yellow_{YEAR}-{m:02d}.parquet"
            if not part.exists():
                print(f"downloading {part.name} ...", flush=True)
                urllib.request.urlretrieve(TLC_URL.format(year=YEAR, month=m), part)
            tbl = pq.read_table(part)
            if schema is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(out, schema)
            # same column names across 2023 months; cast so int/float drift between months cannot break the concat
            tbl = tbl.select(schema.names).cast(schema, safe=False)
            writer.write_table(tbl)
    finally:
        if writer:
            writer.close()
    return out


REGISTRY = {"nyc_taxi": fetch_nyc_taxi}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=REGISTRY)
    p.add_argument("--months", type=int, default=1)
    p.add_argument("--dest", type=Path, default=DATA_DIR)
    a = p.parse_args()
    path = REGISTRY[a.name](a.months, a.dest)
    import pyarrow.parquet as pq

    print(f"{path}  rows={pq.ParquetFile(path).metadata.num_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
