import glob
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ==========================================
csv_files = [f for f in glob.glob('*.csv') if f != 'cleaned_transactions.csv']
if not csv_files:
    raise FileNotFoundError("Please upload your Kaggle CSV file to the Google Colab Files sidebar on the left!")

file_name = csv_files[0]
print(f"Loading dataset: {file_name}\n")
df = pd.read_csv(file_name)

# Clean any accidental spaces in column names
df.columns = df.columns.str.strip()

# Check the number of rows and columns
print("=== 1. DATASET SHAPE ===")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")

# Identify column names and data types
print("=== 2. COLUMN NAMES AND DATA TYPES ===")
print(df.dtypes, "\n")

# Check missing values and duplicate records
print("=== 3. MISSING VALUES PER COLUMN ===")
print(df.isnull().sum(), "\n")

duplicate_count = df.duplicated(subset=['Transaction ID']).sum()
print(f"=== 4. DUPLICATE TRANSACTION IDs: {duplicate_count} ===\n")

# Review basic statistical information
print("=== 5. SUMMARY STATISTICS ===")
print(df.describe(include='all'), "\n")

# ==========================================
# STEP 2: DATA CLEANING & PREPARATION
# ==========================================
# 1. Remove duplicate records where required
df = df.drop_duplicates(subset=['Transaction ID']).copy()

# 2. Correct data types where necessary
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Amount (INR)'] = pd.to_numeric(df['Amount (INR)'], errors='coerce')

# 3. Handle missing values appropriately
df = df.dropna(subset=['Transaction ID', 'Timestamp', 'Amount (INR)', 'Status'])
df['Sender Name'] = df['Sender Name'].fillna('Unknown Sender')
df['Receiver Name'] = df['Receiver Name'].fillna('Unknown Receiver')
df['Sender UPI ID'] = df['Sender UPI ID'].fillna('unknown@upi')
df['Receiver UPI ID'] = df['Receiver UPI ID'].fillna('unknown@upi')

# 4. Check and standardize inconsistent values and categories
df['Status'] = df['Status'].astype(str).str.strip().str.title()

# 5. Create useful calculated fields/features
df['Sender Handle'] = df['Sender UPI ID'].astype(str).apply(
    lambda x: '@' + x.split('@')[1].strip().lower() if '@' in x else '@unknown'
)
df['Receiver Handle'] = df['Receiver UPI ID'].astype(str).apply(
    lambda x: '@' + x.split('@')[1].strip().lower() if '@' in x else '@unknown'
)

df['Date'] = df['Timestamp'].dt.date
df['Hour'] = df['Timestamp'].dt.hour
df['Day of Week'] = df['Timestamp'].dt.day_name()

df['Transaction Tier'] = pd.cut(
    df['Amount (INR)'],
    bins=[-1, 500, 2000, 10000, float('inf')],
    labels=['Micro (< ₹500)', 'Small (₹500 - ₹2,000)', 'Medium (₹2,000 - ₹10,000)', 'High (> ₹10,000)']
)

df['is_failed'] = np.where(df['Status'].str.upper().isin(['FAILED', 'FAILURE', 'FAIL']), 1, 0)

# Save cleaned dataset for the Streamlit Dashboard
df.to_csv('cleaned_transactions.csv', index=False)
print("=== DATA CLEANING COMPLETE: 'cleaned_transactions.csv' created! ===\n")

# Print quick KPI summary for your Word Report
print("=== KEY NUMBERS FOR YOUR REPORT ===")
print(f"Total Cleaned Transactions: {len(df):,}")
print(f"Total Transaction Value (INR): ₹{df['Amount (INR)'].sum():,.2f}")
print(f"Average Transaction Value (INR): ₹{df['Amount (INR)'].mean():,.2f}")
print(f"Overall Failure Rate: {df['is_failed'].mean() * 100:.2f}%\n")

# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# 1. Univariate Analysis: Distribution of Transaction Values (INR)
fig_uni = px.histogram(
    df, x='Amount (INR)', nbins=40, color='Status',
    title='Univariate Analysis: Distribution of Transaction Values (INR) by Status',
    labels={'Amount (INR)': 'Transaction Value (INR)', 'count': 'Number of Transactions'}
)
fig_uni.show()

# 2. Bivariate Analysis: Transaction Volume & Failure Rate by Sender UPI Handle
handle_summary = df.groupby('Sender Handle', as_index=False).agg(
    total_volume=('Transaction ID', 'count'),
    total_value=('Amount (INR)', 'sum'),
    failure_rate=('is_failed', 'mean')
)
handle_summary['failure_rate_pct'] = handle_summary['failure_rate'] * 100

fig_bi = px.bar(
    handle_summary, x='Sender Handle', y='total_volume',
    color='failure_rate_pct',
    title='Bivariate Analysis: Transaction Volume and Failure Rate (%) by UPI Handle',
    labels={
        'Sender Handle': 'Sender UPI Gateway Handle',
        'total_volume': 'Transaction Volume (Count)',
        'failure_rate_pct': 'Failure Rate (%)'
    }
)
fig_bi.show()

# 3. Multivariate Analysis: Failure Rate (%) by Day of Week and UPI Handle
multi_summary = df.groupby(['Day of Week', 'Sender Handle'], as_index=False).agg(
    failure_rate=('is_failed', 'mean'),
    tx_count=('Transaction ID', 'count')
)
multi_summary['failure_rate_pct'] = multi_summary['failure_rate'] * 100

fig_multi = px.density_heatmap(
    multi_summary, x='Sender Handle', y='Day of Week', z='failure_rate_pct',
    histfunc='avg',
    title='Multivariate Analysis: Failure Rate (%) by UPI Gateway Handle and Day of Week',
    labels={
        'Sender Handle': 'Sender UPI Gateway Handle',
        'Day of Week': 'Day of Week',
        'failure_rate_pct': 'Failure Rate (%)'
    }
)
fig_multi.show()

# 4. Correlation Matrix
print("=== CORRELATION MATRIX ===")
print(df[['Amount (INR)', 'Hour', 'is_failed']].corr())
