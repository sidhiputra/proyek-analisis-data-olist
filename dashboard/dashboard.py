import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Konfigurasi halaman dasar Streamlit
st.set_page_config(
    page_title="E-Commerce Public Dashboard - Olist",
    page_icon="🛒",
    layout="wide"
)

sns.set(style="whitegrid")

# ==========================================
# HELPER FUNCTIONS (Penyiapan DataFrame)
# ==========================================
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule="D", on="order_purchase_timestamp").agg({
        "order_id": "nunique",
        "price": "sum",
        "total_price": "sum"
    }).reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "product_revenue",
        "total_price": "total_revenue"
    }, inplace=True)
    return daily_orders_df

def create_monthly_orders_df(df):
    monthly_df = df.resample(rule="ME", on="order_purchase_timestamp").agg({
        "order_id": "nunique",
        "price": "sum"
    }).reset_index()
    monthly_df["year_month"] = monthly_df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    monthly_df.rename(columns={
        "order_id": "total_orders",
        "price": "product_revenue"
    }, inplace=True)
    return monthly_df

def create_category_orders_df(df):
    category_df = df.groupby(by="product_category_name_english", as_index=False).agg({
        "order_item_id": "count",
        "price": "sum"
    })
    category_df.rename(columns={
        "order_item_id": "items_sold",
        "price": "total_revenue"
    }, inplace=True)
    return category_df

def create_state_customers_df(df):
    state_df = df.groupby(by="customer_state", as_index=False).agg({
        "customer_unique_id": "nunique",
        "order_id": "nunique",
        "price": "sum"
    })
    state_df.rename(columns={
        "customer_unique_id": "total_customers",
        "order_id": "total_orders",
        "price": "total_revenue"
    }, inplace=True)
    return state_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_unique_id", "max_order_timestamp", "frequency", "monetary"]
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    reference_date = df["order_purchase_timestamp"].dt.date.max() + pd.Timedelta(days=1)
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (reference_date - x).days)
    rfm_df["short_id"] = rfm_df["customer_unique_id"].str[:8]
    
    # Perhitungan RFM Score & Segmentasi Pelanggan (Manual Grouping)
    rfm_df["r_rank"] = rfm_df["recency"].rank(ascending=False)
    rfm_df["f_rank"] = rfm_df["frequency"].rank(ascending=True)
    rfm_df["m_rank"] = rfm_df["monetary"].rank(ascending=True)

    rfm_df["r_rank_norm"] = (rfm_df["r_rank"] / rfm_df["r_rank"].max()) * 100
    rfm_df["f_rank_norm"] = (rfm_df["f_rank"] / rfm_df["f_rank"].max()) * 100
    rfm_df["m_rank_norm"] = (rfm_df["m_rank"] / rfm_df["m_rank"].max()) * 100

    rfm_df["RFM_score"] = (
        0.15 * rfm_df["r_rank_norm"] + 
        0.28 * rfm_df["f_rank_norm"] + 
        0.57 * rfm_df["m_rank_norm"]
    ) * 0.05
    rfm_df["RFM_score"] = rfm_df["RFM_score"].round(2)

    rfm_df["customer_segment"] = np.where(
        rfm_df["RFM_score"] > 4.5, "Top Customers",
        np.where(
            rfm_df["RFM_score"] > 4.0, "High Value Customers",
            np.where(
                rfm_df["RFM_score"] > 3.0, "Medium Value Customers",
                np.where(rfm_df["RFM_score"] > 1.6, "Low Value Customers", "Lost Customers")
            )
        )
    )
    return rfm_df

# ==========================================
# LOAD & PREPARE DATA
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "main_data.csv")
    df = pd.read_csv(csv_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

all_df = load_data()

min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

# ==========================================
# SIDEBAR (FILTER INTERAKTIF)
# ==========================================
with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.markdown("**Filter Analisis Data**")
    
    date_range = st.date_input(
        label="Pilih Rentang Waktu Transaksi",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Filter tambahan berdasarkan Negara Bagian (State)
    all_states = sorted(all_df["customer_state"].unique().tolist())
    selected_states = st.multiselect(
        label="Filter Negara Bagian (State)",
        options=all_states,
        default=all_states
    )
    
    st.markdown("---")
    st.caption("Proyek Akhir Analisis Data\n**I Putu Weda Sidhi Putra**")

# Memfilter DataFrame utama berdasarkan input pengguna
main_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date) &
    (all_df["customer_state"].isin(selected_states if selected_states else all_states))
]

# Memanggil Helper Functions
daily_orders_df = create_daily_orders_df(main_df)
monthly_orders_df = create_monthly_orders_df(main_df)
category_orders_df = create_category_orders_df(main_df)
state_customers_df = create_state_customers_df(main_df)
rfm_df = create_rfm_df(main_df)

# ==========================================
# HALAMAN UTAMA DASHBOARD
# ==========================================
st.title("📊 E-Commerce Public Dataset Dashboard (Olist)")
st.markdown(f"Menampilkan performa transaksi berstatus **Delivered** pada periode **{start_date}** hingga **{end_date}**.")

# 1. KEY PERFORMANCE INDICATORS (Sweet Spot Kiri Atas)
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_orders = daily_orders_df["order_count"].sum()
    st.metric("Total Pesanan (Orders)", value=f"{total_orders:,}")

with col2:
    total_items = len(main_df)
    st.metric("Total Item Terjual", value=f"{total_items:,}")

with col3:
    product_rev = daily_orders_df["product_revenue"].sum()
    st.metric("Total Revenue Produk", value=f"BRL {product_rev:,.2f}")

with col4:
    unique_cust = main_df["customer_unique_id"].nunique()
    st.metric("Pelanggan Unik Aktif", value=f"{unique_cust:,}")

st.markdown("---")

# 2. VISUALISASI PERTANYAAN 1: TREN PESANAN & PENDAPATAN
st.subheader("📈 Tren Jumlah Pesanan & Pendapatan (Pertanyaan 1)")

tab1, tab2 = st.tabs(["Tren Bulanan (Monthly)", "Tren Harian (Daily)"])

with tab1:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True)
    ax[0].plot(monthly_orders_df["year_month"], monthly_orders_df["total_orders"], marker="o", linewidth=2.5, color="#1E88E5")
    ax[0].set_title("Tren Jumlah Pesanan Bulanan", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Jumlah Pesanan")
    
    ax[1].plot(monthly_orders_df["year_month"], monthly_orders_df["product_revenue"], marker="s", linewidth=2.5, color="#2A9D8F")
    ax[1].set_title("Tren Pendapatan Produk Bulanan (BRL)", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Revenue (BRL)")
    ax[1].set_xlabel("Periode (Tahun-Bulan)")
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(daily_orders_df["order_purchase_timestamp"], daily_orders_df["order_count"], linewidth=1.5, color="#1E88E5")
    ax.set_title("Pergerakan Jumlah Pesanan Harian", fontsize=14, fontweight="bold")
    ax.set_ylabel("Jumlah Pesanan")
    ax.set_xlabel("Tanggal")
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# 3. VISUALISASI PERTANYAAN 2: PERFORMA KATEGORI PRODUK & DEMOGRAFI WILAYAH
st.subheader("🏆 Performa Kategori Produk & Sebaran Demografi Pelanggan (Pertanyaan 2)")

col_left, col_right = st.columns(2)
colors_highlight = ["#1E88E5", "#B0BEC5", "#B0BEC5", "#B0BEC5", "#B0BEC5"]

with col_left:
    fig, ax = plt.subplots(figsize=(8, 5))
    top5_cat = category_orders_df.sort_values(by="items_sold", ascending=False).head(5)
    sns.barplot(
        x="items_sold", y="product_category_name_english", hue="product_category_name_english",
        data=top5_cat, palette=colors_highlight, legend=False, ax=ax
    )
    ax.set_title("Top 5 Kategori Produk Paling Laris", fontsize=13, fontweight="bold")
    ax.set_xlabel("Jumlah Item Terjual")
    ax.set_ylabel(None)
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9, fontweight="bold")
    if not top5_cat.empty:
        ax.set_xlim(0, top5_cat["items_sold"].max() * 1.18)
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom5_cat = category_orders_df.sort_values(by="items_sold", ascending=True).head(5)
    sns.barplot(
        x="items_sold", y="product_category_name_english", hue="product_category_name_english",
        data=bottom5_cat, palette=colors_highlight, legend=False, ax=ax
    )
    ax.set_title("Bottom 5 Kategori Produk Paling Sedikit Terjual", fontsize=13, fontweight="bold")
    ax.set_xlabel("Jumlah Item Terjual")
    ax.set_ylabel(None)
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9, fontweight="bold")
    if not bottom5_cat.empty:
        ax.set_xlim(0, bottom5_cat["items_sold"].max() * 1.2)
    plt.tight_layout()
    st.pyplot(fig)

# Grafik Demografi Negara Bagian (State)
fig, ax = plt.subplots(figsize=(14, 5))
top10_state = state_customers_df.sort_values(by="total_customers", ascending=False).head(10)
colors_state = ["#1E88E5"] + ["#B0BEC5"] * (len(top10_state) - 1) if len(top10_state) > 0 else ["#1E88E5"]
sns.barplot(
    x="total_customers", y="customer_state", hue="customer_state",
    data=top10_state, palette=colors_state, legend=False, ax=ax
)
ax.set_title("Top 10 Negara Bagian (State) Berdasarkan Jumlah Pelanggan Unik", fontsize=14, fontweight="bold")
ax.set_xlabel("Jumlah Pelanggan Unik")
ax.set_ylabel("Kode Negara Bagian (State)")
for container in ax.containers:
    ax.bar_label(container, padding=4, fontsize=9, fontweight="bold")
if not top10_state.empty:
    ax.set_xlim(0, top10_state["total_customers"].max() * 1.15)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 4. ANALISIS LANJUTAN: RFM ANALYSIS & CUSTOMER SEGMENTATION
st.subheader("💎 Analisis Lanjutan: RFM Parameters & Customer Segmentation")

r_col, f_col, m_col = st.columns(3)
with r_col:
    avg_recency = round(rfm_df["recency"].mean(), 1) if not rfm_df.empty else 0
    st.metric("Average Recency (Days)", value=f"{avg_recency} Hari")
with f_col:
    avg_freq = round(rfm_df["frequency"].mean(), 2) if not rfm_df.empty else 0
    st.metric("Average Frequency (Orders)", value=f"{avg_freq} Kali")
with m_col:
    avg_mon = rfm_df["monetary"].mean() if not rfm_df.empty else 0
    st.metric("Average Monetary", value=f"BRL {avg_mon:,.2f}")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5.5))
colors_rfm = ["#1E88E5"] * 5

if not rfm_df.empty:
    top5_r = rfm_df.sort_values(by="recency", ascending=True).head(5)
    sns.barplot(y="recency", x="short_id", hue="short_id", data=top5_r, palette=colors_rfm[:len(top5_r)], legend=False, ax=ax[0])
    ax[0].set_title("Top 5 By Recency (Days)", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("Customer ID (8-Digit Prefix)")
    ax[0].set_ylabel("Recency (Days)")
    for container in ax[0].containers:
        ax[0].bar_label(container, padding=3, fontsize=9, fontweight="bold")

    top5_f = rfm_df.sort_values(by="frequency", ascending=False).head(5)
    sns.barplot(y="frequency", x="short_id", hue="short_id", data=top5_f, palette=colors_rfm[:len(top5_f)], legend=False, ax=ax[1])
    ax[1].set_title("Top 5 By Frequency (Orders)", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("Customer ID (8-Digit Prefix)")
    ax[1].set_ylabel("Total Orders")
    for container in ax[1].containers:
        ax[1].bar_label(container, padding=3, fontsize=9, fontweight="bold")

    top5_m = rfm_df.sort_values(by="monetary", ascending=False).head(5)
    sns.barplot(y="monetary", x="short_id", hue="short_id", data=top5_m, palette=colors_rfm[:len(top5_m)], legend=False, ax=ax[2])
    ax[2].set_title("Top 5 By Monetary (BRL)", fontsize=13, fontweight="bold")
    ax[2].set_xlabel("Customer ID (8-Digit Prefix)")
    ax[2].set_ylabel("Total Spend (BRL)")
    for container in ax[2].containers:
        ax[2].bar_label(container, fmt="%.0f", padding=3, fontsize=9, fontweight="bold")

plt.tight_layout()
st.pyplot(fig)

# Grafik Segmentasi Pelanggan berdasarkan RFM Score
segment_counts_df = rfm_df.groupby(by="customer_segment", as_index=False, observed=False).agg({"customer_unique_id": "count"})
segment_counts_df.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)
segment_order = ["Top Customers", "High Value Customers", "Medium Value Customers", "Low Value Customers", "Lost Customers"]
segment_counts_df["customer_segment"] = pd.Categorical(segment_counts_df["customer_segment"], categories=segment_order, ordered=True)
segment_counts_df.sort_values(by="customer_segment", inplace=True)

fig, ax = plt.subplots(figsize=(12, 4.5))
colors_seg = ["#B0BEC5", "#B0BEC5", "#B0BEC5", "#1E88E5", "#B0BEC5"]
sns.barplot(
    x="customer_count", y="customer_segment", hue="customer_segment",
    data=segment_counts_df, palette=colors_seg, legend=False, ax=ax
)
ax.set_title("Distribusi Segmen Pelanggan Berdasarkan RFM Score (Manual Grouping)", fontsize=14, fontweight="bold")
ax.set_xlabel("Jumlah Pelanggan Unik")
ax.set_ylabel("Segmen Pelanggan")
for container in ax.containers:
    ax.bar_label(container, padding=4, fontsize=9, fontweight="bold")
if not segment_counts_df.empty:
    ax.set_xlim(0, segment_counts_df["customer_count"].max() * 1.15)
plt.tight_layout()
st.pyplot(fig)

st.caption("Copyright © 2026 - I Putu Weda Sidhi Putra | Proyek Analisis Data Dicoding")