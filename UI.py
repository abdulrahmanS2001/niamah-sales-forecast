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

# Import Babel for Arabic number formatting
try:
    from babel.numbers import format_decimal
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Babel"])
    from babel.numbers import format_decimal

# Arabic number formatting helper
def fmt_num(x):
    try:
        x = float(x)
        if abs(x) < 1:
            return format_decimal(round(x, 1), locale='ar')  # e.g., ٠٫٨
        return format_decimal(int(round(x)), locale='ar')   # integers with Arabic digits and grouping
    except Exception:
        return str(x)

# Global column header translation mapping
COLUMN_TRANSLATIONS = {
    "Branch": "الفرع",
    "Description": "الوصف", 
    "Month": "الشهر",
    "Month ": "الشهر",  # Handle the space variant
    "Qty": "الكمية"
}

def translate_columns(df):
    """Translate DataFrame column headers to Arabic while keeping data unchanged."""
    df_display = df.copy()
    df_display.columns = [COLUMN_TRANSLATIONS.get(col, col) for col in df_display.columns]
    return df_display

st.set_page_config(page_title="منصّة التنبؤ بالمبيعات", layout="wide")

# Inject CSS for RTL and Arabic font
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Tajawal', sans-serif !important;
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# Title with Arabic-only support
st.title("منصّة التنبؤ بالمبيعات")

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
st.sidebar.header("البيانات")
use_bundled = st.sidebar.checkbox("استخدام الملف المرفق (niamah_sales.csv)", value=True)
uploaded = None if use_bundled else st.sidebar.file_uploader("رفع ملف CSV", type=["csv"])

# Determine data path
if use_bundled:
    data_path = "niamah_sales.csv"
    if not os.path.exists(data_path):
        st.error("Bundled file `niamah_sales.csv` not found.")
        st.stop()
else:
    if uploaded is None:
        st.info("ارفع ملف CSV أو فعّل 'استخدام الملف المرفق'.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        data_path = tmp.name

# Load and clean data
with st.spinner("تحميل وتنظيف البيانات..."):
    df_clean, diagnostics = load_and_clean_data(data_path)

if df_clean.empty:
    st.error("لم يتم العثور على بيانات صالحة بعد التنظيف. يرجى التحقق من تنسيق ملف CSV.")
    st.stop()

# Branch selection
st.subheader("اختيار الفرع")
branches = sorted(df_clean['Branch'].unique().tolist())
branch_index = 0

# Reset product selection if branch changes
if st.session_state.selected_branch is not None:
    try:
        branch_index = branches.index(st.session_state.selected_branch)
    except ValueError:
        branch_index = 0

selected_branch = st.selectbox(
    "الفرع", 
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
st.subheader("اختيار المنتج")

# Get products for selected branch
available_products = get_branch_products(df_clean, selected_branch)

if not available_products:
    st.warning(f"لم يتم العثور على منتجات للفرع: {selected_branch}")
    st.stop()

# Search box
search_term = st.text_input(
    "ابحث عن منتج", 
    value=st.session_state.product_search,
    placeholder="اكتب جزءًا من اسم المنتج…",
    key="product_search_input"
)

# Update session state
if search_term != st.session_state.product_search:
    st.session_state.product_search = search_term

# Filter products based on search
filtered_products = filter_products_by_search(available_products, search_term)

if not filtered_products:
    st.warning(f"لا توجد منتجات تطابق مصطلح البحث: '{search_term}'")
    st.info(f"عدد المنتجات المتاحة: {fmt_num(len(available_products))}")
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
    f"المنتج (الوصف) — عدد المطابقات: {fmt_num(len(filtered_products))}",
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
    st.error("لم يتم العثور على بيانات لمجموعة الفرع/المنتج المحددة.")
    st.stop()

# Display cleaned data
st.subheader("البيانات المنظّفة (مدخل النموذج)")
st.dataframe(translate_columns(df_filtered), use_container_width=True)

# Forecast range setup
if df_filtered['Month '].isna().all():
    st.error("لم يتم العثور على بيانات شهرية صالحة بعد التصفية.")
    st.stop()

try:
    m_last = int(df_filtered['Month '].dropna().max())
except (ValueError, TypeError):
    st.error("بيانات شهرية غير صالحة. يرجى التحقق من تنسيق ملف CSV.")
    st.stop()

default_from = m_last + 1
default_to = max(default_from, m_last + 2)

# Forecast range inputs
st.subheader("نطاق التنبؤ")
c3, c4 = st.columns(2)
with c3:
    m_from = st.number_input(
        "التنبؤ من (إزاحة الشهر)", 
        min_value=0, 
        value=default_from, 
        step=1
    )
with c4:
    m_to = st.number_input(
        "التنبؤ إلى (إزاحة الشهر، شاملة)", 
        min_value=int(m_from), 
        value=int(max(default_to, m_from)), 
        step=1
    )

# Run forecast
run_forecast = st.button("تنفيذ التنبؤ", type="primary")

if run_forecast:
    with st.spinner("تشغيل نموذج التنبؤ..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
            img_path = img_tmp.name
        
        try:
            # Run the regression analysis using existing reg_anal function
            summary = reg_anal(df_filtered, [int(m_from), int(m_to)], img_path)
            
            # Display results
            st.subheader("النتائج")
            
            # Extract values for explanation
            beta1 = float(summary.loc["Beta 1", "value"])
            conf = summary.loc["95% Confidence Interval", "value"]
            preds = summary.loc["Predicted Values", "value"]
            pred_int = summary.loc["Prediction Interval", "value"]
            
            # Parse predictions for individual months
            try:
                pred_list = eval(str(preds)) if isinstance(preds, str) else preds
                pred1 = float(pred_list[0]) if len(pred_list) > 0 else 0
                pred2 = float(pred_list[1]) if len(pred_list) > 1 else 0
            except:
                pred1 = pred2 = 0
            
            # Parse prediction interval
            try:
                pi_parts = str(pred_int).replace('[', '').replace(']', '').split(',')
                pi_low = float(pi_parts[0].strip())
                pi_high = float(pi_parts[1].strip())
            except:
                pi_low = pi_high = 0
            
            # Determine trend
            trend_word = "ارتفاع" if beta1 > 0 else "انخفاض"
            trend_arrow = "⬆️" if beta1 > 0 else "⬇️"
            
            # Summary cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("الاتجاه العام", f"{trend_word} {trend_arrow}")
            with col2:
                st.metric("توقع الشهر القادم", fmt_num(pred1))
            with col3:
                st.metric("نطاق الثقة ٩٥٪ (إجمالي الفترة)", f"{fmt_num(pi_low)} – {fmt_num(pi_high)}")
            
            # Arabic summary
            st.info(
                f"الخلاصة: الاتجاه **{trend_word}**. متوقع بيع {fmt_num(pred1)} وحدة في الشهر القادم، ثم {fmt_num(pred2)} وحدة بعده. "
                f"نطاق الثقة ٩٥٪ لإجمالي الفترة: من {fmt_num(pi_low)} إلى {fmt_num(pi_high)}."
            )
            
            # Optional advanced mode
            adv = st.toggle("وضع متقدّم", value=False)
            if adv:
                # Format summary table with Arabic numbers
                summary_formatted = summary.copy()
                for idx in summary_formatted.index:
                    val = summary_formatted.loc[idx, "value"]
                    try:
                        if isinstance(val, (int, float)):
                            summary_formatted.loc[idx, "value"] = fmt_num(val)
                        elif isinstance(val, str) and '[' in val:
                            # Handle lists/intervals
                            parts = val.replace('[', '').replace(']', '').split(',')
                            formatted_parts = [fmt_num(float(p.strip())) for p in parts]
                            summary_formatted.loc[idx, "value"] = '[' + ', '.join(formatted_parts) + ']'
                    except:
                        pass
                st.table(translate_columns(summary_formatted))
            
            # Display visualization
            st.subheader("التصور")
            st.image(img_path, caption="اتجاه المبيعات والتوقّع", use_container_width=True)
            
            # Download buttons
            st.subheader("التحميلات")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_buf = io.StringIO()
                summary.to_csv(csv_buf)
                st.download_button(
                    "تنزيل الملخّص",
                    data=csv_buf.getvalue(),
                    file_name="summary.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Image download
                with open(img_path, "rb") as fh:
                    st.download_button(
                        "تنزيل الرسم",
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