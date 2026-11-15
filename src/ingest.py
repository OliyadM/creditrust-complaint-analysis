"""Download the CFPB narratives archive and filter it to CrediTrust's products.

CFPB stopped shipping complaint narratives in the live bulk download, but the
full records (Dec 2011 - Aug 2026) are kept in the FOIA reading room as 21 zip
files. This script downloads them, streams each CSV in chunks, keeps rows with
a non-empty narrative that map to one of the five product lines and writes:

  data/filtered/filtered_complaints.csv.gz   every matching complaint
  data/processed/sample_complaints.csv.gz    seeded per-product sample
  reports/product_counts.csv                 counts per product

Usage:
  python -m src.ingest download
  python -m src.ingest filter
  python -m src.ingest sample --per-product 4000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src import config
from src.products import PRODUCTS, clean_narrative, map_products_frame

ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
ZIP_RE = re.compile(r"https?://files\.consumerfinance\.gov/f/documents/CCDB_Export_[^\"']+\.zip")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

USECOLS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company", "State", "Complaint ID",
]
RENAME = {
    "Date received": "date_received",
    "Product": "cfpb_product",
    "Sub-product": "cfpb_sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Complaint ID": "complaint_id",
}


# File names as listed on the archive page in Nov 2026. Used when the page
# itself refuses the request (it sits behind a bot filter that sometimes
# rejects python clients even though the files download fine).
FILE_BASE = "https://files.consumerfinance.gov/f/documents/"
KNOWN_FILES = [
    "CCDB_Export_1_December_2011_through_April_2018.zip",
    "CCDB_Export_2_May_2018_through_April_2021.zip",
    "CCDB_Export_3_May_2021_through_October_2022.zip",
    "CCDB_Export_4_November_2022_through_August_2023.zip",
    "CCDB_Export_5_September_2023_through_March_2024.zip",
    "CCDB_Export_6_April_2024_through_July_2024.zip",
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]


def list_archive_zips(html: str) -> list[str]:
    return sorted(set(ZIP_RE.findall(html)))


def download(dest: Path = config.RAW_DIR) -> list[Path]:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    try:
        urls = list_archive_zips(requests.get(ARCHIVE_PAGE, headers=HEADERS, timeout=60).text)
    except requests.RequestException:
        urls = []
    if not urls:
        print("archive page not readable, falling back to the known file list")
        urls = [FILE_BASE + name for name in KNOWN_FILES]
    print(f"{len(urls)} archive files")
    paths = []
    for url in urls:
        out = dest / url.rsplit("/", 1)[-1]
        paths.append(out)
        if out.exists():
            continue
        print("downloading", out.name)
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = out.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
            tmp.rename(out)
    return paths


def filter_zip(path: Path, chunksize: int = 200_000):
    """Yield filtered chunks from one archive zip."""
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as fh:
        for chunk in pd.read_csv(fh, usecols=USECOLS, dtype=str, chunksize=chunksize):
            n_rows = len(chunk)
            chunk = chunk[chunk["Consumer complaint narrative"].fillna("").str.strip() != ""]
            n_narr = len(chunk)
            chunk = chunk.assign(product=map_products_frame(chunk))
            chunk = chunk[chunk["product"].notna()].rename(columns=RENAME)
            yield n_rows, n_narr, chunk


def run_filter(raw_dir: Path = config.RAW_DIR, out: Path = config.FILTERED_PATH) -> pd.DataFrame:
    zips = sorted(raw_dir.glob("CCDB_Export_*.zip"))
    if not zips:
        sys.exit(f"No archive zips in {raw_dir}; run `python -m src.ingest download` first.")
    out.parent.mkdir(parents=True, exist_ok=True)
    total = with_narr = 0
    parts = []
    for z in zips:
        for n_rows, n_narr, chunk in filter_zip(z):
            total += n_rows
            with_narr += n_narr
            parts.append(chunk)
        print(f"{z.name}: running total {total:,} rows, {with_narr:,} with narrative")
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("complaint_id")
    df = df.sort_values("date_received").reset_index(drop=True)
    df.to_csv(out, index=False, compression="gzip")

    counts = df["product"].value_counts().reindex(PRODUCTS, fill_value=0)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    counts.rename_axis("product").rename("complaints").to_csv(config.REPORTS_DIR / "product_counts.csv")
    summary = {
        "archive_files": len(zips),
        "rows_read": total,
        "rows_with_narrative": with_narr,
        "in_scope_rows": before,
        "in_scope_unique": len(df),
        "date_min": df["date_received"].min(),
        "date_max": df["date_received"].max(),
        "per_product": counts.to_dict(),
    }
    (config.REPORTS_DIR / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return df


def stratified_sample(df: pd.DataFrame, per_product: int, seed: int = config.SEED) -> pd.DataFrame:
    """Take up to `per_product` complaints from each product with a fixed seed."""
    parts = [
        g.sample(n=min(per_product, len(g)), random_state=seed)
        for _, g in df.groupby("product", sort=True)
    ]
    out = pd.concat(parts).sort_values("complaint_id").reset_index(drop=True)
    out["clean_narrative"] = out["narrative"].map(clean_narrative)
    return out[out["clean_narrative"].str.len() > 0].reset_index(drop=True)


def run_sample(per_product: int, src: Path = config.FILTERED_PATH, out: Path = config.SAMPLE_PATH):
    df = pd.read_csv(src, dtype=str)
    sample = stratified_sample(df, per_product)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, compression="gzip")
    print(sample["product"].value_counts().to_string())
    return sample


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("filter")
    s = sub.add_parser("sample")
    s.add_argument("--per-product", type=int, default=4000)
    args = ap.parse_args(argv)
    if args.cmd == "download":
        download()
    elif args.cmd == "filter":
        run_filter()
    else:
        run_sample(args.per_product)


if __name__ == "__main__":
    main()
