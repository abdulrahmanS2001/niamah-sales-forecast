# UI.py
# Refactored Streamlit UI for Niamah Sales Forecasting with enhanced features

import os
import io
import sys
import tempfile
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Auto-install dependencies if missing
try:
    import streamlit as st
except ModuleNotFoundError:
    import subprocess
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user", "--upgrade",
            "streamlit", "pandas", "numpy", "statsmodels", "matplotlib", "scipy"
        ])
    except subprocess.CalledProcessError:
        print("Installation failed. Please run this command manually in PowerShell as Administrator:")
        print("C:\\Python312\\python.exe -m pip install streamlit pandas numpy statsmodels matplotlib scipy")
        sys.exit(1)
    
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Installation completed but module still not found.")
        print("Please restart your terminal and run the script again.")
        sys.exit(1)

from utils import normalize_txt, safe_int_convert, parse_month_with_fallback
from reg_anal import reg_anal

st.set_page_config(page_title="Niamah Sales Forecast (نعمة للتنبؤ بالمبيعات)", layout="wide")

# Title with bilingual support
st.title("Niamah Sales Forecast (نعمة للتنبؤ بالمبيعات)")
st.markdown("*Poisson GLM-based forecasting for branch-product sales*")

# Initialize session state
if 'selected_branch' not in st.session_state:
    st.session_state.selected_branch = None
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'product_search' not in st.session_state:
    st.session_state.product_search = ""

@st.cache_data
def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load CSV and perform initial cleaning with diagnostic counts.
    Returns cleaned dataframe and diagnostic information.
    """
    try:
        raw_df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame(), {}
    
    diagnostics = {}
    diagnostics['total_rows'] = len(raw_df)
    
    # Handle column name variations (Month vs Month )
    month_col = None
    for col in ['Month ', 'Month']:
        if col in raw_df.columns:
            month_col = col
            break
    
    if month_col is None:
        st.error("CSV must contain a 'Month' or 'Month ' column")
        return pd.DataFrame(), diagnostics
    
    # Normalize the month column name to 'Month'
    if month_col != 'Month':
        raw_df = raw_df.rename(columns={month_col: 'Month'})
    
    # Check required columns
    required_cols = {'Branch', 'Description', 'Qty', 'Month'}
    missing_cols = required_cols - set(raw_df.columns)
    if missing_cols:
        st.error(f"CSV missing required columns: {missing_cols}")
        return pd.DataFrame(), diagnostics
    
    # Step 1: Normalize text fields
    raw_df['Branch_norm'] = raw_df['Branch'].astype(str).apply(normalize_txt)
    raw_df['Description_norm'] = raw_df['Description'].astype(str).apply(normalize_txt)
    
    # Remove rows with empty normalized values
    df = raw_df[
        (raw_df['Branch_norm'] != '') & 
        (raw_df['Description_norm'] != '')
    ].copy()
    diagnostics['after_text_cleaning'] = len(df)
    
    # Step 2: Convert quantities to positive integers
    df['Qty_clean'] = df['Qty'].apply(safe_int_convert)
    df = df[df['Qty_clean'] > 0].copy()
    diagnostics['after_qty_cleaning'] = len(df)
    
    # Step 3: Parse months to offsets
    df['MonthOffset'] = df['Month'].apply(lambda x: parse_month_with_fallback(x, ("2024", "01")))
    
    # Separate valid and invalid month parsing
    valid_months = df['MonthOffset'].notna()
    df_valid = df[valid_months].copy()
    df_invalid = df[~valid_months].copy()
    
    diagnostics['after_month_parsing'] = len(df_valid)
    diagnostics['failed_month_samples'] = df_invalid['Month'].unique()[:10].tolist() if len(df_invalid) > 0 else []
    
    # Step 4: Aggregate by Branch, Description, and MonthOffset
    if len(df_valid) > 0:
        df_agg = df_valid.groupby(['Branch_norm', 'Description_norm', 'MonthOffset']).agg({
            'Qty_clean': 'sum'
        }).reset_index()
        
        # Rename columns for consistency with existing code
        df_agg = df_agg.rename(columns={
            'Branch_norm': 'Branch',
            'Description_norm': 'Description', 
            'Qty_clean': 'Qty',
            'MonthOffset': 'Month '
        })
        
        diagnostics['final_aggregated_rows'] = len(df_agg)
    else:
        df_agg = pd.DataFrame()
        diagnostics['final_aggregated_rows'] = 0
    
    return df_agg, diagnostics

@st.cache_data
def get_branch_products(df: pd.DataFrame, branch: str) -> List[str]:
    """Get list of products available for a specific branch."""
    if df.empty:
        return []
    
    branch_data = df[df['Branch'] == branch]
    return sorted(branch_data['Description'].unique().tolist())

def filter_products_by_search(products: List[str], search_term: str) -> List[str]:
    """Filter products based on normalized search term."""
    if not search_term:
        return products
    
    search_norm = normalize_txt(search_term.lower())
    filtered = []
    
    for product in products:
        product_norm = normalize_txt(product.lower())
        if search_norm in product_norm:
            filtered.append(product)
    
    return filtered

# Sidebar: Data source
st.sidebar.header("Data (البيانات)")
use_bundled = st.sidebar.checkbox("Use bundled niamah_sales.csv (استخدام الملف المرفق)", value=True)
uploaded = None if use_bundled else st.sidebar.file_uploader("Upload CSV (رفع ملف)", type=["csv"])

# Determine data path
if use_bundled:
    data_path = "niamah_sales.csv"
    if not os.path.exists(data_path):
        st.error("Bundled file `niamah_sales.csv` not found.")
        st.stop()
else:
    if uploaded is None:
        st.info("Upload your CSV or enable 'Use bundled niamah_sales.csv'.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        data_path = tmp.name

# Load and clean data
with st.spinner("Loading and cleaning data... (تحميل وتنظيف البيانات...)"):
    df_clean, diagnostics = load_and_clean_data(data_path)

if df_clean.empty:
    st.error("No valid data found after cleaning. Please check your CSV format.")
    st.stop()

# Branch selection
st.subheader("Branch Selection (اختيار الفرع)")
branches = sorted(df_clean['Branch'].unique().tolist())
branch_index = 0

# Reset product selection if branch changes
if st.session_state.selected_branch is not None:
    try:
        branch_index = branches.index(st.session_state.selected_branch)
    except ValueError:
        branch_index = 0

selected_branch = st.selectbox(
    "Branch (الفرع)", 
    branches, 
    index=branch_index,
    key="branch_selector"
)

# Check if branch changed and reset product selection
if selected_branch != st.session_state.selected_branch:
    st.session_state.selected_branch = selected_branch
    st.session_state.selected_product = None
    st.session_state.product_search = ""
    st.rerun()

# Product selection with search
st.subheader("Product Selection (اختيار المنتج)")

# Get products for selected branch
available_products = get_branch_products(df_clean, selected_branch)

if not available_products:
    st.warning(f"No products found for branch: {selected_branch}")
    st.stop()

# Search box
search_term = st.text_input(
    "Search product (ابحث عن منتج)", 
    value=st.session_state.product_search,
    placeholder="Type part of product name... (اكتب جزء من اسم المنتج...)",
    key="product_search_input"
)

# Update session state
if search_term != st.session_state.product_search:
    st.session_state.product_search = search_term

# Filter products based on search
filtered_products = filter_products_by_search(available_products, search_term)

if not filtered_products:
    st.warning(f"No products match search term: '{search_term}'")
    st.info(f"Available products count: {len(available_products)}")
    st.stop()

# Product selection
product_index = 0
if (st.session_state.selected_product is not None and 
    st.session_state.selected_product in filtered_products):
    try:
        product_index = filtered_products.index(st.session_state.selected_product)
    except ValueError:
        product_index = 0

selected_product = st.selectbox(
    f"Product (Description) (المنتج - الوصف) - {len(filtered_products)} matches",
    filtered_products,
    index=product_index,
    key="product_selector"
)

st.session_state.selected_product = selected_product

# Filter data for selected branch and product
df_filtered = df_clean[
    (df_clean['Branch'] == selected_branch) & 
    (df_clean['Description'] == selected_product)
].copy()

if df_filtered.empty:
    st.error("No data found for the selected Branch/Product combination.")
    st.stop()

# Display cleaned data
st.subheader("Cleaned Data (model input) (البيانات المنظفة - مدخل النموذج)")
st.dataframe(df_filtered, use_container_width=True)

# Forecast range setup
if df_filtered['Month '].isna().all():
    st.error("No valid month data found after filtering.")
    st.stop()

try:
    m_last = int(df_filtered['Month '].dropna().max())
except (ValueError, TypeError):
    st.error("Invalid month data. Please check your CSV format.")
    st.stop()

default_from = m_last + 1
default_to = max(default_from, m_last + 2)

# Forecast range inputs
st.subheader("Forecast Range (نطاق التنبؤ)")
c3, c4 = st.columns(2)
with c3:
    m_from = st.number_input(
        "Forecast from (month offset) (التنبؤ من - إزاحة الشهر)", 
        min_value=0, 
        value=default_from, 
        step=1
    )
with c4:
    m_to = st.number_input(
        "Forecast to (month offset, inclusive) (التنبؤ إلى - إزاحة الشهر شاملة)", 
        min_value=int(m_from), 
        value=int(max(default_to, m_from)), 
        step=1
    )

# Run forecast
run_forecast = st.button("🔮 Run Forecast (تنفيذ التنبؤ)", type="primary")

if run_forecast:
    with st.spinner("Running Poisson GLM forecast... (تشغيل نموذج التنبؤ...)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
            img_path = img_tmp.name
        
        try:
            # Run the regression analysis using existing reg_anal function
            summary = reg_anal(df_filtered, [int(m_from), int(m_to)], img_path)
            
            # Display results
            st.subheader("📊 Results (النتائج)")
            st.table(summary)
            
            # Extract values for explanation
            beta1 = summary.loc["Beta 1", "value"]
            conf = summary.loc["95% Confidence Interval", "value"]
            preds = summary.loc["Predicted Values", "value"]
            pred_int = summary.loc["Prediction Interval", "value"]
            
            # Bilingual explanation
            st.info(
                f"**English:** Predicted monthly change for {selected_product} at {selected_branch}: {beta1}. "
                f"95% confidence interval for slope: {conf}. "
                f"Predicted values for months [{int(m_from)}, {int(m_to)}]: {preds}. "
                f"95% prediction interval for total: {pred_int}.\n\n"
                f"**العربية:** التغيير الشهري المتوقع لـ {selected_product} في فرع {selected_branch}: {beta1}. "
                f"فترة الثقة 95% للميل: {conf}. "
                f"القيم المتوقعة للأشهر [{int(m_from)}, {int(m_to)}]: {preds}. "
                f"فترة التنبؤ 95% للإجمالي: {pred_int}."
            )
            
            # Display visualization
            st.subheader("📈 Visualization (التصور)")
            st.image(img_path, caption="Sales Trend & Forecast (Poisson GLM)", use_container_width=True)
            
            # Download buttons
            st.subheader("💾 Downloads (التحميلات)")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_buf = io.StringIO()
                summary.to_csv(csv_buf)
                st.download_button(
                    "📄 Download summary.csv",
                    data=csv_buf.getvalue(),
                    file_name="summary.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Image download
                with open(img_path, "rb") as fh:
                    st.download_button(
                        "🖼️ Download prediction.jpg",
                        data=fh.read(),
                        file_name="prediction.jpg",
                        mime="image/jpeg"
                    )
                        
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.exception(e)
        finally:
            # Cleanup temp files
            if not use_bundled and os.path.exists(data_path):
                try:
                    os.remove(data_path)
                except:
                    pass

# Footer
st.markdown("---")
st.markdown(
    "*Built with Streamlit • Enhanced with Arabic text normalization and robust month parsing*"
)