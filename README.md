# Niamah Sales Forecast (نعمة للتنبؤ بالمبيعات)

A sophisticated Streamlit application for sales forecasting using Poisson GLM regression, enhanced with Arabic text normalization and robust data processing.

## Features (الميزات)

### 🚀 **Enhanced UI Features**
- **Branch-filtered Products**: Product dropdown shows only items available for selected branch
- **Smart Search**: Arabic/English-aware search with normalization
- **Session State Management**: Remembers selections and resets appropriately
- **Bilingual Interface**: English labels with Arabic translations
- **Real-time Diagnostics**: Shows data cleaning steps and statistics

### 🔤 **Text Processing**
- **Unicode Normalization**: NFKC normalization for consistent text
- **Arabic Digit Conversion**: ٠١٢٣٤٥٦٧٨٩ → 0123456789
- **Whitespace Handling**: NBSP removal and space normalization
- **Case-insensitive Search**: Works across Arabic and English text

### 📅 **Month Parsing**
- **Multiple Formats**: Supports "2025- Feb", "Jan", "Aug-2024", "2024/07", etc.
- **Robust Fallback**: Generic parser with legacy mapping support
- **Base Date Configurable**: Default 2024-01 offset system
- **Error Reporting**: Shows unparseable month samples in diagnostics

### 📊 **Analytics**
- **Poisson GLM Regression**: Advanced statistical modeling
- **Confidence Intervals**: 95% CI for slope parameters
- **Prediction Intervals**: Uncertainty quantification for forecasts
- **Data Aggregation**: Intelligent grouping by branch-product-month

## Installation (التثبيت)

### Prerequisites (المتطلبات المسبقة)
- Python 3.8+ (بايثون 3.8 أو أحدث)
- pip package manager (مدير الحزم pip)

### Quick Start (البداية السريعة)
```bash
# Clone or download the project
# استنساخ أو تحميل المشروع

# Navigate to project directory
# الانتقال إلى مجلد المشروع
cd NiamahStats-main

# Install dependencies (will auto-install if missing)
# تثبيت التبعيات (سيتم التثبيت التلقائي إذا كانت مفقودة)
pip install streamlit pandas numpy statsmodels matplotlib scipy

# Run the application
# تشغيل التطبيق
streamlit run UI.py
```

### Alternative Installation (التثبيت البديل)
The app will automatically install missing dependencies on first run.

## Usage (الاستخدام)

### 1. **Data Input (إدخال البيانات)**
- Use bundled `niamah_sales.csv` or upload your own
- Required columns: `Branch`, `Description`, `Qty`, `Month`/`Month `

### 2. **Branch Selection (اختيار الفرع)**
- Select from available branches
- Product list updates automatically

### 3. **Product Search & Selection (البحث واختيار المنتج)**
- Type part of product name in Arabic or English
- Search is normalization-aware and case-insensitive
- Select from filtered results

### 4. **Forecasting (التنبؤ)**
- Set forecast range (month offsets from 2024-01)
- Click "Run Forecast" to generate predictions
- View results, visualization, and download outputs

### 5. **Diagnostics (التشخيص)**
- Expand diagnostics section to see:
  - Data cleaning step counts
  - Branch breakdown
  - Failed month parsing samples
  - Current selection summary

## File Structure (هيكل الملفات)

```
NiamahStats-main/
├── UI.py                 # Main Streamlit application
├── utils.py              # Utility functions (normalization, parsing)
├── reg_anal.py          # Regression analysis (Poisson GLM)
├── data_prep.py         # Legacy data preparation (kept for compatibility)
├── main.py              # Original CLI interface
├── niamah_sales.csv     # Sample data
├── tests/               # Unit tests
│   ├── __init__.py
│   ├── test_normalize.py
│   ├── test_month_parser.py
│   └── test_branch_product_filter.py
└── README.md            # This file
```

## Testing (الاختبار)

Run the comprehensive test suite:

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_normalize.py -v
pytest tests/test_month_parser.py -v
pytest tests/test_branch_product_filter.py -v

# Quick test run
pytest tests/ -q
```

### Test Coverage (تغطية الاختبارات)
- **Text Normalization**: Unicode, Arabic digits, whitespace handling
- **Month Parsing**: Multiple formats, edge cases, error handling
- **Branch-Product Filtering**: Data integrity, search functionality, aggregation

## Technical Details (التفاصيل التقنية)

### Data Processing Pipeline (خط معالجة البيانات)
1. **Load CSV** → Raw data import
2. **Text Normalization** → Clean branch/product names
3. **Quantity Filtering** → Remove zero/negative quantities
4. **Month Parsing** → Convert to numeric offsets
5. **Aggregation** → Group by branch-product-month
6. **Caching** → Store processed data for performance

### Month Offset System (نظام إزاحة الأشهر)
- **Base**: 2024-01 = offset 0
- **Example**: 2025-02 = offset 13 (13 months after base)
- **Supports**: Past and future dates relative to base

### Normalization Functions (وظائف التطبيع)
- `normalize_txt()`: Comprehensive text cleaning
- `safe_int_convert()`: Robust quantity parsing
- `parse_month_with_fallback()`: Multi-format month parsing

## Configuration (التكوين)

### Customization Options (خيارات التخصيص)
- **Base Date**: Change in `utils.py` → `parse_month_with_fallback()`
- **UI Language**: Modify labels in `UI.py`
- **Caching**: Adjust `@st.cache_data` parameters
- **Month Formats**: Extend patterns in `parse_month_generic()`

### Performance Tuning (ضبط الأداء)
- Data caching enabled for files up to ~10K rows
- Session state management for UI responsiveness
- Lazy loading of branch-product combinations

## Troubleshooting (استكشاف الأخطاء)

### Common Issues (المشاكل الشائعة)

**1. "No valid data found after cleaning"**
- Check CSV format and required columns
- Verify month formats are supported
- Ensure quantities are numeric and positive

**2. "No products found for branch"**
- Check branch name spelling and normalization
- Verify data contains the selected branch
- Review diagnostics for data cleaning steps

**3. Month parsing failures**
- Check diagnostics for failed month samples
- Extend month patterns in `utils.py` if needed
- Verify date formats match supported patterns

**4. Search not working**
- Ensure search terms match product names
- Try partial matches (Arabic text is normalized)
- Check for typos in Arabic text

### Performance Issues (مشاكل الأداء)
- Large CSV files (>50K rows) may need caching adjustments
- Clear Streamlit cache: `streamlit cache clear`
- Restart app if session state becomes corrupted

## Contributing (المساهمة)

### Development Setup (إعداد التطوير)
```bash
# Install development dependencies
pip install pytest black flake8

# Run code formatting
black *.py tests/

# Run linting
flake8 *.py tests/

# Run tests
pytest tests/ -v
```

### Adding Features (إضافة الميزات)
1. Add utility functions to `utils.py`
2. Write comprehensive tests
3. Update UI components in `UI.py`
4. Document changes in README

## License (الترخيص)

This project is provided as-is for educational and commercial use.

## Support (الدعم)

For technical support or feature requests:
- Review diagnostics in the app
- Check test cases for expected behavior
- Refer to code comments for implementation details

---

*Built with ❤️ using Streamlit • Enhanced Arabic text processing • Robust month parsing*

*تم البناء بـ ❤️ باستخدام Streamlit • معالجة محسنة للنص العربي • تحليل قوي للأشهر*
