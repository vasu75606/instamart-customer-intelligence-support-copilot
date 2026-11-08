from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"


def audit():
    customers = pd.read_csv(RAW / "customers.csv", dtype=str, keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)
    tickets = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)

    print("RAW TABLE SHAPES")
    for name, frame in [("customers", customers), ("products", products), ("orders", orders), ("tickets", tickets)]:
        print(f"\n{name}: shape={frame.shape}")
        print("dtypes:")
        print(frame.dtypes.to_string())
        missing = frame.isna().sum()
        # Empty strings are the source's blanks; report those as missing for string fields.
        blank = frame.astype("string").apply(lambda s: s.str.strip().eq("")).sum()
        counts = missing.add(blank, fill_value=0).astype(int)
        report = pd.DataFrame({"missing_count": counts, "missing_pct": (counts / len(frame) * 100).round(2)})
        print(report.to_string())

    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    audit_stats = {
        "raw orders rows": len(orders),
        "exact duplicate order rows": int(orders.duplicated().sum()),
        "pending rows": int(orders["status"].str.lower().eq("pending").sum()),
        "distinct order_id": int(orders["order_id"].nunique()),
        "non-numeric order_amount": int(amount_num.isna().sum()),
        "negative delivery_minutes": int(pd.to_numeric(orders["delivery_minutes"], errors="coerce").lt(0).sum()),
        "blank delivery_minutes": int(orders["delivery_minutes"].str.strip().eq("").sum()),
        "blank customer age": int(customers["age"].str.strip().eq("").sum()),
        "age outside 10-100": int(age_num.notna().mul(~age_num.between(10, 100)).sum()),
        "distinct raw city spellings": int(customers["city"].nunique()),
        "distinct nonblank raw gender spellings": int(customers.loc[customers.gender.str.strip().ne(""), "gender"].nunique()),
        "blank gender": int(customers["gender"].str.strip().eq("").sum()),
    }
    print("\nAUDIT CHECKPOINTS")
    for k, v in audit_stats.items():
        print(f"{k}: {v}")
    return customers, products, orders, audit_stats


def clean_customers(customers):
    df = customers.copy()
    df["city"] = df["city"].astype(str).str.strip().str.title()
    gender = df["gender"].astype(str).str.strip().str.lower()
    df["gender"] = gender.map({"male": "male", "m": "male", "female": "female", "f": "female"}).fillna("unknown")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(10, 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df.groupby("city")["age"].transform("median"))
    df["age"] = df["age"].round().astype(int)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="raise")
    return df


def clean_orders(orders, products):
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="raise")
    before_exact = len(df)
    df = df.drop_duplicates().copy()
    exact_removed = before_exact - len(df)

    before_latest = len(df)
    df = df.sort_values(["order_id", "order_ts"], kind="mergesort")
    df = df.drop_duplicates(subset=["order_id"], keep="last").copy()
    multi_removed = before_latest - len(df)

    for col in ["quantity", "discount_pct", "delivery_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    product_lookup = products.set_index("product_id")["price_inr"]
    df["price_inr"] = df["product_id"].map(product_lookup)
    missing_amount = df["order_amount"].isna()
    df.loc[missing_amount, "order_amount"] = (
        df.loc[missing_amount, "price_inr"] * df.loc[missing_amount, "quantity"] *
        (1 - df.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)
    negative_delivery_count = int(df["delivery_minutes"].lt(0).sum())
    df["delivery_minutes"] = df["delivery_minutes"].mask(df["delivery_minutes"] < 0)
    completed = df["status"].eq("completed")
    completed_missing_before_impute = int((completed & df["delivery_minutes"].isna()).sum())
    category_map = products.set_index("product_id")["category"]
    df["category"] = df["product_id"].map(category_map)
    medians = df.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    df.loc[completed, "delivery_minutes"] = df.loc[completed, "delivery_minutes"].fillna(medians)
    df.loc[~completed, "delivery_minutes"] = np.nan
    df["order_amount"] = df["order_amount"].round(2)
    df["quantity"] = df["quantity"].astype(int)
    df["discount_pct"] = df["discount_pct"].astype(int)
    df["delivery_minutes"] = df["delivery_minutes"].round().astype("Int64")
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(columns=["price_inr", "category"])
    changes = {"exact_duplicate_rows_removed": exact_removed, "older_duplicate_order_records_removed": multi_removed,
               "missing_amounts_reconstructed": int(missing_amount.sum()),
               "negative_delivery_times_set_missing": negative_delivery_count,
               "completed_delivery_times_imputed": completed_missing_before_impute}
    return df.sort_values("order_id").reset_index(drop=True), changes


def build_database(customers, products, orders):
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)


def validate(customers, products, orders):
    checks = {
        "customers_count": len(customers) == 1200,
        "six_cities": customers.city.nunique() == 6,
        "no_missing_customer_ages": customers.age.isna().sum() == 0,
        "gender_counts": customers.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
        "products_count": len(products) == 40,
        "orders_count": len(orders) == 12309,
        "no_pending_orders": not orders.status.eq("pending").any(),
        "order_ids_unique": orders.order_id.is_unique,
        "completed_count": int(orders.status.eq("completed").sum()) == 11110,
        "cancelled_count": int(orders.status.eq("cancelled").sum()) == 1199,
        "no_missing_amounts": orders.order_amount.isna().sum() == 0,
        "completed_delivery_filled": orders.loc[orders.status.eq("completed"), "delivery_minutes"].notna().all(),
        "cancelled_delivery_missing": orders.loc[orders.status.eq("cancelled"), "delivery_minutes"].isna().all(),
        "completed_revenue": round(orders.loc[orders.status.eq("completed"), "order_amount"].sum(), 2) == 2998668.80,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("Failed checkpoints: " + ", ".join(failed))
    print("ALL CHECKPOINTS PASSED")


def main():
    raw_customers, raw_products, raw_orders, _ = audit()
    products = raw_products.copy()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    customers = clean_customers(raw_customers)
    orders, changes = clean_orders(raw_orders, products)
    print("\nCLEANING CHANGES")
    print(f"Customer ages imputed/cleaned: {int(pd.to_numeric(raw_customers.age, errors='coerce').isna().sum()) + int((pd.to_numeric(raw_customers.age, errors='coerce').dropna().pipe(lambda s: ~s.between(10, 100))).sum())}")
    print(changes)
    print("\nCUSTOMER CITY COUNTS\n", customers.city.value_counts().to_string())
    print("\nCUSTOMER GENDER COUNTS\n", customers.gender.value_counts().to_string())
    print("\nORDER CHECKPOINTS")
    print("rows:", len(orders))
    print("status counts:\n", orders.status.value_counts().to_string())
    print("missing amounts:", orders.order_amount.isna().sum())
    print("completed missing delivery:", orders.loc[orders.status.eq("completed"), "delivery_minutes"].isna().sum())
    print("cancelled missing delivery:", orders.loc[orders.status.eq("cancelled"), "delivery_minutes"].isna().sum())
    print("completed revenue INR:", round(orders.loc[orders.status.eq("completed"), "order_amount"].sum(), 2))
    build_database(customers, products, orders)
    validate(customers, products, orders)

if __name__ == "__main__":
    main()
