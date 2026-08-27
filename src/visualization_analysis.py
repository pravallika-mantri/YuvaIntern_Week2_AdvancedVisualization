import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# WEEK 2: ADVANCED DATA VISUALIZATION AND STORYTELLING
# Online Retail Dataset
# ============================================================

# -----------------------------
# 1. Setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_online_retail.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 70)
print("WEEK 2 - ADVANCED DATA VISUALIZATION AND STORYTELLING")
print("=" * 70)

# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded successfully.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# 3. Prepare Data
# -----------------------------

# Convert date column if available
date_column = None

for col in ["InvoiceDate", "Invoice Date", "invoicedate", "invoice_date"]:
    if col in df.columns:
        date_column = col
        break

if date_column is None:
    raise ValueError("InvoiceDate column was not found.")

df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

# Identify important columns
quantity_column = None
price_column = None
country_column = None
customer_column = None
invoice_column = None
stock_column = None

for col in df.columns:
    col_lower = col.lower().replace(" ", "").replace("_", "")

    if col_lower == "quantity":
        quantity_column = col
    elif col_lower in ["unitprice", "price"]:
        price_column = col
    elif col_lower == "country":
        country_column = col
    elif col_lower in ["customerid", "customer"]:
        customer_column = col
    elif col_lower in ["invoiceno", "invoice"]:
        invoice_column = col
    elif col_lower in ["stockcode", "productcode"]:
        stock_column = col

# Revenue calculation
if "Revenue" in df.columns:
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
elif quantity_column and price_column:
    df[quantity_column] = pd.to_numeric(df[quantity_column], errors="coerce")
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")
    df["Revenue"] = df[quantity_column] * df[price_column]
else:
    raise ValueError("Unable to identify Quantity and UnitPrice columns.")

# Remove invalid dates/revenue
df = df.dropna(subset=[date_column, "Revenue"])

# Keep valid sales
df = df[df["Revenue"] > 0].copy()

# Time features
df["Year"] = df[date_column].dt.year
df["Month"] = df[date_column].dt.month
df["Month_Name"] = df[date_column].dt.strftime("%b")
df["Year_Month"] = df[date_column].dt.to_period("M").astype(str)

print(f"\nPrepared dataset rows: {len(df):,}")

# ============================================================
# VISUALIZATION 1
# Monthly Revenue and Order Trend
# ============================================================

monthly = df.groupby("Year_Month").agg(
    Revenue=("Revenue", "sum"),
    Orders=(invoice_column, "nunique") if invoice_column else ("Revenue", "count")
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(
    monthly["Year_Month"],
    monthly["Revenue"],
    marker="o",
    linewidth=2,
    label="Revenue"
)

ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue")
ax1.tick_params(axis="x", rotation=60)

# Highlight highest revenue month
max_idx = monthly["Revenue"].idxmax()
max_month = monthly.loc[max_idx, "Year_Month"]
max_revenue = monthly.loc[max_idx, "Revenue"]

ax1.annotate(
    f"Peak: {max_month}\n£{max_revenue:,.0f}",
    xy=(max_idx, max_revenue),
    xytext=(max_idx - 3, max_revenue * 0.85),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10
)

plt.title("Monthly Revenue Trend with Peak Performance")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_monthly_revenue_story.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# VISUALIZATION 2
# Revenue Contribution by Country
# ============================================================

country_revenue = (
    df.groupby(country_column)["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(
    country_revenue.index,
    country_revenue.values
)

ax.set_xlabel("Revenue")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries by Revenue Contribution")

for bar, value in zip(bars, country_revenue.values):
    ax.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f" £{value:,.0f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_country_revenue_story.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# VISUALIZATION 3
# Customer Value Segmentation - Revenue Contribution
# ============================================================

if customer_column:

    customer_value = (
        df.groupby(customer_column)["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    # Divide customers into three equal-sized value groups
    customer_segments = pd.qcut(
        customer_value,
        q=3,
        labels=["Low Value", "Medium Value", "High Value"],
        duplicates="drop"
    )

    # Create customer segment summary
    segment_summary = pd.DataFrame({
        "CustomerValue": customer_value,
        "Segment": customer_segments
    })

    segment_revenue = (
        segment_summary.groupby("Segment", observed=True)["CustomerValue"]
        .sum()
        .reindex(["Low Value", "Medium Value", "High Value"])
    )

    total_segment_revenue = segment_revenue.sum()

    revenue_percent = (
        segment_revenue / total_segment_revenue * 100
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.bar(
        segment_revenue.index,
        segment_revenue.values
    )

    ax.set_title("Customer Value Segmentation: Revenue Contribution")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Total Revenue (£)")

    # Add revenue and percentage labels
    for bar, revenue, percentage in zip(
        bars,
        segment_revenue.values,
        revenue_percent.values
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"£{revenue:,.0f}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_customer_value_segments.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

else:
    print(
        "\nCustomer ID column not available. "
        "Skipping customer segmentation."
    )

# ============================================================
# VISUALIZATION 4
# Product Revenue vs Quantity
# ============================================================

if stock_column and quantity_column:

    product_data = (
        df.groupby(stock_column)
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=(quantity_column, "sum")
        )
        .reset_index()
    )

    # Select top products by revenue
    top_products = product_data.nlargest(30, "Revenue")

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        top_products["Quantity"],
        top_products["Revenue"],
        s=80,
        alpha=0.7
    )

    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Revenue")
    ax.set_title("Top Products: Revenue vs Quantity Sold")

    # Annotate highest revenue product
    highest = top_products.iloc[0]

    ax.annotate(
        "Highest Revenue Product",
        xy=(highest["Quantity"], highest["Revenue"]),
        xytext=(highest["Quantity"], highest["Revenue"] * 0.8),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "04_product_revenue_vs_quantity.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

else:
    print("\nProduct or Quantity column not available. Skipping product analysis.")

# ============================================================
# VISUALIZATION 5
# Monthly Seasonal Heatmap
# ============================================================

monthly_pivot = (
    df.groupby(["Year", "Month"])["Revenue"]
    .sum()
    .unstack()
)

monthly_pivot = monthly_pivot.reindex(
    columns=range(1, 13)
)

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly_pivot.columns = month_labels

fig, ax = plt.subplots(figsize=(14, 7))

sns.heatmap(
    monthly_pivot,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Seasonal Revenue Pattern by Year and Month")
ax.set_xlabel("Month")
ax.set_ylabel("Year")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_seasonal_revenue_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# VISUALIZATION 6
# Revenue Distribution - Focused View with Outliers
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Calculate the 99th percentile
upper_limit = df["Revenue"].quantile(0.99)

# Separate normal transactions and extreme outliers
main_revenue = df[df["Revenue"] <= upper_limit]["Revenue"]
outlier_count = (df["Revenue"] > upper_limit).sum()

# Plot the main distribution
sns.histplot(
    main_revenue,
    bins=50,
    kde=True,
    ax=ax
)

# Calculate median
median_revenue = df["Revenue"].median()

# Median line
ax.axvline(
    median_revenue,
    linestyle="--",
    linewidth=2,
    label=f"Median: £{median_revenue:.2f}"
)

# 99th percentile line
ax.axvline(
    upper_limit,
    linestyle=":",
    linewidth=2,
    label=f"99th Percentile: £{upper_limit:,.2f}"
)

ax.set_title(
    "Transaction Revenue Distribution (99th Percentile Focus)"
)

ax.set_xlabel("Revenue per Transaction (£)")
ax.set_ylabel("Number of Transactions")

# Explain the outliers directly on the chart
ax.text(
    0.98,
    0.95,
    f"Extreme transactions excluded from main view: {outlier_count:,}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10
)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_revenue_distribution_story.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 2 VISUALIZATION SUMMARY")
print("=" * 70)

print(f"\nTotal revenue analysed: £{df['Revenue'].sum():,.2f}")
print(f"Total quantity sold: {df[quantity_column].sum():,.0f}" if quantity_column else "")
print(f"Unique products: {df[stock_column].nunique():,}" if stock_column else "")
print(f"Unique customers: {df[customer_column].nunique():,}" if customer_column else "")
print(f"Highest revenue month: {max_month}")
print(f"Highest monthly revenue: £{max_revenue:,.2f}")

print("\nVisualizations created:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.endswith(".png"):
        print(f"  - {filename}")

print("\nAll visualizations saved successfully.")
print("=" * 70)
print("WEEK 2 ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)