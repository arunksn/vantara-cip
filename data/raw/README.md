# data/raw

Original, unmodified source files go here. Nothing is committed (see `.gitignore`).

* **Option B (default):** `python -m src.pipeline` downloads the dataset through `ucimlrepo` (UCI id 502) and caches it as
  `online_retail_II_ucimlrepo.csv.gz` in this folder.
* **Option A (fallback):** download `online_retail_II.xlsx` from the UCI repository
  (https://archive.ics.uci.edu/dataset/502/online+retail+ii), place it here as `online_retail_II.xlsx`; it is used
  automatically when the download is unavailable.
