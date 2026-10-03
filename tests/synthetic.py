"""Synthetic stand-in for Online Retail II (SAME SCHEMA, NOT REAL DATA).

Used only for unit tests and for verifying the pipeline where the UCI download is unavailable.
Replicates the documented data-quality issues: missing Customer ID, returns/cancellations, zero prices,
admin StockCodes, exact duplicates, inconsistent descriptions and quantity outliers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START = pd.Timestamp("2009-12-01")
END = pd.Timestamp("2011-12-09 20:00:00")

NOUNS = {
    "lighting": ["LIGHT", "LAMP", "CANDLE", "LANTERN"],
    "kitchen_dining": ["MUG", "TEAPOT", "PLATE", "BOWL", "SPOON", "JAR"],
    "bags_storage": ["BAG", "BOX", "BASKET", "TOTE"],
    "christmas_seasonal": ["CHRISTMAS DECORATION", "XMAS STAR", "SANTA FIGURE", "ADVENT CALENDAR"],
    "stationery_cards": ["CARD", "NOTEBOOK", "PENCIL", "STICKER SET", "WRAP"],
    "toys_games": ["TOY", "GAME", "PUZZLE", "DOLL"],
    "garden_outdoor": ["GARDEN TROWEL", "BIRD FEEDER", "FLOWER POT", "WINDMILL"],
    "jewelry_accessories": ["NECKLACE", "BRACELET", "EARRING", "SCARF"],
    "bath_beauty": ["SOAP", "BATH SPONGE", "TOWEL"],
    "home_decor": ["PHOTO FRAME", "CUSHION COVER", "CLOCK", "MIRROR", "VASE"],
    "other": ["WIDGET", "ORNAMENT", "TRINKET", "CHARM"],
}
ADJ = ["VINTAGE", "RETROSPOT", "PINK", "BLUE", "RED", "GREEN", "SMALL", "LARGE", "WOODEN", "FLORAL", "SPOTTY", "IVORY"]
COUNTRIES = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal"]
ADMIN = [("POST", 18.0), ("DOT", 25.0), ("M", 12.0), ("BANK CHARGES", 15.0)]


def make_synthetic_transactions(n_customers: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Generate a raw transaction table with the original UCI column names."""
    rng = np.random.default_rng(seed)
    cats = list(NOUNS)
    k = len(cats)

    # Product catalogue
    prod_rows = []
    for ci, cat in enumerate(cats):
        for j in range(max(40, 300 // k * 2)):
            noun = NOUNS[cat][j % len(NOUNS[cat])]
            desc = f"{ADJ[j % len(ADJ)]} {noun} {j // len(ADJ) + 1}"
            prod_rows.append((f"{20000 + ci * 1000 + j}", desc, float(np.round(rng.lognormal(0.8, 0.6), 2)), ci))
    prods = pd.DataFrame(prod_rows, columns=["stock_code", "desc", "price", "cat"])
    cat_idx = [np.flatnonzero(prods["cat"].to_numpy() == ci) for ci in range(k)]

    total_days = (END - START).days
    cust_ids = np.arange(12347, 12347 + n_customers)
    prefs = rng.dirichlet(np.ones(k) * 0.35, size=n_customers)
    seasonal = rng.random(n_customers) < 0.18
    rate = rng.lognormal(np.log(1 / 50), 0.8, n_customers)
    loyal = rng.random(n_customers) < 0.30
    life = np.where(loyal, 5000.0, rng.exponential(260.0, n_customers) + 30)
    declining = rng.random(n_customers) < 0.35
    bulk = rng.random(n_customers) < 0.10
    ret_rate = np.clip(rng.beta(1, 18, n_customers), 0, 0.5)
    disc_p = rng.beta(1, 5, n_customers)
    first = rng.beta(1.0, 1.7, n_customers) * (total_days - 20)
    country = rng.choice(COUNTRIES, n_customers, p=[0.86, 0.03, 0.025, 0.02, 0.015, 0.015, 0.015, 0.01, 0.01])

    order_cust, order_t = [], []
    for i in range(n_customers):
        t_end = min(first[i] + life[i], total_days)
        span = t_end - first[i]
        if span <= 0:
            continue
        lam = rate[i] * (4.0 if seasonal[i] else 1.5)
        n = rng.poisson(lam * span)
        t = np.sort(first[i] + rng.random(n + 1) * span)
        t = np.concatenate([[first[i]], t])
        month = ((START + pd.to_timedelta(t, unit="D")).month).to_numpy()
        q4 = month >= 10
        accept = np.where(seasonal[i], np.where(q4, 1.0, 0.04), np.where(q4, 1.0, 0.65))
        if declining[i]:
            accept = accept * np.clip(1.0 - 0.8 * (t - first[i]) / max(span, 1), 0.1, 1)
        keep = rng.random(len(t)) < accept
        keep[0] = True
        t = t[keep]
        order_cust.extend([i] * len(t))
        order_t.extend(t.tolist())

    orders = pd.DataFrame({"c": order_cust, "t": order_t}).sort_values("t").reset_index(drop=True)
    orders["inv_no"] = 489434 + np.arange(len(orders))
    orders["guest"] = rng.random(len(orders)) < 0.0
    n_lines = np.clip(rng.geometric(1 / 9, len(orders)), 1, 60)
    guest_orders = pd.DataFrame({"c": -1, "t": rng.random(int(len(orders) * 0.25)) * total_days})
    guest_orders["inv_no"] = 900000 + np.arange(len(guest_orders))
    guest_orders["guest"] = True
    all_orders = pd.concat([orders.assign(n=n_lines), guest_orders.assign(n=np.clip(rng.geometric(1 / 8, len(guest_orders)), 1, 40))],
                           ignore_index=True)

    rep = np.repeat(np.arange(len(all_orders)), all_orders["n"].to_numpy())
    lines = pd.DataFrame({"order": rep})
    lines["c"] = all_orders["c"].to_numpy()[rep]
    lines["t"] = all_orders["t"].to_numpy()[rep]
    lines["inv_no"] = all_orders["inv_no"].to_numpy()[rep]
    lines["guest"] = all_orders["guest"].to_numpy()[rep]
    cidx = np.where(lines["c"].to_numpy() >= 0, lines["c"].to_numpy(), 0)
    cum = np.cumsum(prefs[cidx], axis=1)
    cat_pick = np.minimum((rng.random(len(lines))[:, None] > cum).sum(1), k - 1)
    prod_pick = np.empty(len(lines), dtype=int)
    for ci in range(k):
        m = cat_pick == ci
        pool = cat_idx[ci]
        prod_pick[m] = pool[np.minimum((rng.random(m.sum()) ** 2 * len(pool)).astype(int), len(pool) - 1)]
    lines["prod"] = prod_pick
    base_qty = rng.choice([1, 2, 3, 4, 6, 8, 12, 24], len(lines), p=[0.2, 0.15, 0.1, 0.1, 0.15, 0.1, 0.12, 0.08])
    lines["qty"] = base_qty * np.where(bulk[cidx] & ~lines["guest"].to_numpy(), rng.choice([1, 5, 10], len(lines)), 1)
    discounted = rng.random(len(lines)) < disc_p[cidx]
    lines["price"] = np.round(prods["price"].to_numpy()[prod_pick] * np.where(discounted, 0.8, 1.0), 2)

    # Cancellations
    rr = ret_rate[cidx]
    cancel_mask = (rng.random(len(lines)) < rr * 0.5) & ~lines["guest"].to_numpy()
    canc = lines[cancel_mask].copy()
    canc["t"] = np.minimum(canc["t"] + rng.integers(1, 30, len(canc)), total_days - 0.01)
    canc["qty"] = -np.minimum(canc["qty"], rng.integers(1, 5, len(canc)) * 2)
    canc["inv_no"] = 700000 + canc["order"]
    canc["inv_prefix"] = "C"

    lines["inv_prefix"] = ""
    allx = pd.concat([lines, canc], ignore_index=True)
    allx["Invoice"] = allx["inv_prefix"].fillna("") + allx["inv_no"].astype(str)
    allx["StockCode"] = prods["stock_code"].to_numpy()[allx["prod"].to_numpy()]
    desc = prods["desc"].to_numpy()[allx["prod"].to_numpy()].astype(object)
    noise = rng.random(len(allx))
    desc = np.where(noise < 0.04, np.char.lower(desc.astype(str)), desc)
    desc = np.where((noise >= 0.04) & (noise < 0.07), np.char.add(desc.astype(str), "  "), desc)
    allx["Description"] = desc
    hours = rng.uniform(8, 20, len(allx))
    allx["InvoiceDate"] = START + pd.to_timedelta(allx["t"].to_numpy(), unit="D").floor("D") + pd.to_timedelta(hours, unit="h")
    allx["InvoiceDate"] = allx["InvoiceDate"].clip(upper=END)
    allx["Quantity"] = allx["qty"].astype(int)
    allx["Price"] = allx["price"].astype(float)
    cid = np.where(allx["c"].to_numpy() >= 0, cust_ids[np.where(allx["c"].to_numpy() >= 0, allx["c"].to_numpy(), 0)], np.nan)
    allx["Customer ID"] = cid
    ctry = np.where(allx["c"].to_numpy() >= 0, country[np.where(allx["c"].to_numpy() >= 0, allx["c"].to_numpy(), 0)], "United Kingdom")
    allx["Country"] = ctry
    df = allx[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]].copy()

    # Admin lines
    n_admin = int(len(df) * 0.004)
    adm = df.sample(n_admin, random_state=seed).copy()
    pick = rng.integers(0, len(ADMIN), n_admin)
    adm["StockCode"] = [ADMIN[p][0] for p in pick]
    adm["Description"] = ["POSTAGE" if ADMIN[p][0] == "POST" else "MANUAL" for p in pick]
    adm["Price"] = [ADMIN[p][1] for p in pick]
    adm["Quantity"] = 1

    # Zero-price adjustments (no customer), entry-error pair, exact duplicates
    adj = df.sample(int(len(df) * 0.002), random_state=seed + 1).copy()
    adj["Price"] = 0.0
    adj["Quantity"] = -rng.integers(1, 20, len(adj))
    adj["Customer ID"] = np.nan
    adj["Invoice"] = adj["Invoice"].str.replace("^C", "", regex=True)
    errs = df[df["Customer ID"].notna() & (df["Quantity"] > 0)].sample(3, random_state=seed + 2).copy()
    errs_rev = errs.copy()
    errs["Quantity"] = 80995
    errs_rev["Quantity"] = -80995
    errs_rev["Invoice"] = "C" + errs_rev["Invoice"].astype(str)
    errs_rev["InvoiceDate"] = (errs_rev["InvoiceDate"] + pd.Timedelta(hours=2)).clip(upper=END)
    dups = df.sample(int(len(df) * 0.003), random_state=seed + 3)

    df = pd.concat([df, adm, adj, errs, errs_rev, dups], ignore_index=True)
    df = df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    return df
