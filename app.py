import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="FinTech UPI Payment Transaction Analytics",
    layout="wide"
)

st.title("FinTech Digital Payment (UPI) Transaction Analytics Dashboard")
st.markdown("Interactive analysis of transaction volumes, UPI gateway handles, transaction values (INR), temporal patterns, and failure rates.")

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_transactions.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df = load_data()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Filter Transactions")

selected_status = st.sidebar.multiselect(
    "Transaction Status",
    options=df['Status'].unique(),
    default=df['Status'].unique()
)

selected_handles = st.sidebar.multiselect(
    "Sender UPI Gateway Handle",
    options=df['Sender Handle'].unique(),
    default=df['Sender Handle'].unique()
)

selected_tiers = st.sidebar.multiselect(
    "Transaction Value Tier",
    options=df['Transaction Tier'].dropna().unique(),
    default=df['Transaction Tier'].dropna().unique()
)

selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=df['Day of Week'].unique(),
    default=df['Day of Week'].unique()
)

filtered_df = df[
    (df['Status'].isin(selected_status)) &
    (df['Sender Handle'].isin(selected_handles)) &
    (df['Transaction Tier'].isin(selected_tiers)) &
    (df['Day of Week'].isin(selected_days))
]

# ==========================================
# TOP KPIs
# ==========================================
total_tx = len(filtered_df)
total_val = filtered_df['Amount (INR)'].sum()
avg_val = filtered_df['Amount (INR)'].mean() if total_tx > 0 else 0
failure_rate = (filtered_df['is_failed'].mean() * 100) if total_tx > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Transaction Volume", f"{total_tx:,}")
kpi2.metric("Total Transaction Value (INR)", f"₹{total_val:,.2f}")
kpi3.metric("Average Transaction Value (INR)", f"₹{avg_val:,.2f}")
kpi4.metric("Transaction Failure Rate", f"{failure_rate:.2f}%")

st.markdown("---")

# ==========================================
# CHARTS & VISUALIZATIONS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    # Chart 1: Volume & Value by Sender UPI Handle
    handle_df = filtered_df.groupby('Sender Handle', as_index=False).agg(
        volume=('Transaction ID', 'count'),
        value=('Amount (INR)', 'sum')
    ).sort_values(by='volume', ascending=False)
    fig1 = px.bar(
        handle_df, x='Sender Handle', y='volume', color='Sender Handle',
        title="Transaction Volume by Sender UPI Handle",
        labels={'Sender Handle': 'Sender UPI Handle (Gateway)', 'volume': 'Transaction Volume (Count)'}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Chart 2: Failure Rate by Sender vs Receiver UPI Handle
    fail_handle_df = filtered_df.groupby('Sender Handle', as_index=False)['is_failed'].mean()
    fail_handle_df['failure_rate_pct'] = fail_handle_df['is_failed'] * 100
    fail_handle_df = fail_handle_df.sort_values(by='failure_rate_pct', ascending=False)
    fig2 = px.bar(
        fail_handle_df, x='Sender Handle', y='failure_rate_pct', color='failure_rate_pct',
        title="Transaction Failure Rate (%) by Sender UPI Handle",
        labels={'Sender Handle': 'Sender UPI Handle (Gateway)', 'failure_rate_pct': 'Failure Rate (%)'}
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Chart 3: Hourly Transaction Failure Rate Pattern
    hourly_df = filtered_df.groupby('Hour', as_index=False).agg(
        failure_rate=('is_failed', 'mean'),
        volume=('Transaction ID', 'count')
    )
    hourly_df['failure_rate_pct'] = hourly_df['failure_rate'] * 100
    fig3 = px.line(
        hourly_df, x='Hour', y='failure_rate_pct', markers=True,
        title="Transaction Failure Rate (%) by Hour of Day",
        labels={'Hour': 'Hour of Day (0-23)', 'failure_rate_pct': 'Failure Rate (%)'}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Chart 4: Transaction Value Distribution across Value Tiers
    tier_df = filtered_df.groupby(['Transaction Tier', 'Status'], as_index=False)['Transaction ID'].count()
    fig4 = px.bar(
        tier_df, x='Transaction Tier', y='Transaction ID', color='Status', barmode='group',
        title="Transaction Outcomes Across Value Tiers (INR)",
        labels={'Transaction Tier': 'Transaction Value Tier (INR)', 'Transaction ID': 'Number of Transactions (Count)'}
    )
    st.plotly_chart(fig4, use_container_width=True)
