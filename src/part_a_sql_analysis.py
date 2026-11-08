from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
SQL = ROOT / "sql" / "queries.sql"


def main():
    with sqlite3.connect(DB) as conn:
        statements = [s.strip() for s in SQL.read_text(encoding="utf-8").split(";") if s.strip()]
        print("SQL QUERY RESULTS")
        for i, statement in enumerate(statements, 1):
            result = pd.read_sql_query(statement, conn)
            print(f"Q{i}: rows={len(result)}")
            if i == 2:
                print("Q2 total INR:", round(result.revenue_inr.sum(), 2))
                print("Q2 largest cell:", result.loc[result.revenue_inr.idxmax()].to_dict())
            elif i == 5:
                avg = pd.read_sql_query("SELECT AVG(total_spend) AS avg_spend FROM (SELECT customer_id, SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id)", conn).iloc[0, 0]
                print("Q5 average customer spend INR:", round(avg, 2))
            elif i == 6:
                tops = result.sort_values(["city", "spend_rank", "customer_id"]).groupby("city", as_index=False).first()
                print("Q6 top customer per city:\n", tops[["city", "customer_id", "total_spend"]].to_string(index=False))
                print("Q6 rows where RANK differs from ROW_NUMBER:", int((result.spend_rank != result.spend_row_number).sum()))
            elif i == 7:
                print("Q7 last cumulative INR:", result.cumulative_revenue_inr.iloc[-1])

        orders = pd.read_sql_query("SELECT * FROM orders WHERE status='completed'", conn)
        customers = pd.read_sql_query("SELECT * FROM customers", conn)
        products = pd.read_sql_query("SELECT * FROM products", conn)

    joined = orders.merge(customers[["customer_id", "city"]], on="customer_id").merge(products[["product_id", "category"]], on="product_id")
    pivot = pd.pivot_table(joined, values="order_amount", index="city", columns="category", aggfunc="sum", fill_value=0, margins=True, margins_name="All")
    print("\nPANDAS PIVOT TABLE (INR)\n", pivot.round(2).to_string())
    print("PIVOT GRAND TOTAL INR:", round(float(pivot.loc["All", "All"]), 2))
    city_stats = joined.groupby("city").agg(
        sum_inr=("order_amount", "sum"),
        mean_inr=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    print("\nCITY AGGREGATES\n", city_stats.round(2).to_string())

    amount = orders.order_amount.astype(float)
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print("\nORDER AMOUNT STATISTICS")
    print({"mean": amount.mean(), "median": amount.median(), "skewness": amount.skew(),
           "Q1": q1, "Q2_median": amount.quantile(.5), "Q3": q3,
           "IQR": iqr, "lower_fence": lower, "upper_fence": upper,
           "outliers_below": int((amount < lower).sum()), "outliers_above": int((amount > upper).sum())})
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = orders[corr_cols].corr(numeric_only=True)
    print("\nCORRELATION MATRIX\n", corr.round(4).to_string())
    pairs = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    print("Largest absolute correlation:", pairs.index[0], "r=", corr.loc[pairs.index[0][0], pairs.index[0][1]])

if __name__ == "__main__":
    main()
