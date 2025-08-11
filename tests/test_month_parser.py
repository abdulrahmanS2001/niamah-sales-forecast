import pytest
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import parse_month_generic, parse_month_with_fallback


class TestMonthParser:
    """Test cases for month parsing functions."""
    
    def test_current_format_parsing(self):
        """Test parsing of current CSV format: '2025- Feb', '2024- Jan'."""
        assert parse_month_generic("2025- Feb", ("2024", "01")) == 13  # Feb 2025 = 13 months from Jan 2024
        assert parse_month_generic("2024- Jan", ("2024", "01")) == 0   # Jan 2024 = base month
        assert parse_month_generic("2024- Dec", ("2024", "01")) == 11  # Dec 2024 = 11 months from Jan 2024
        assert parse_month_generic("2025- Jan", ("2024", "01")) == 12  # Jan 2025 = 12 months from Jan 2024
    
    def test_bare_month_names(self):
        """Test parsing of bare month names: 'Feb', 'Jan', 'Mar'."""
        # Bare months should assume appropriate year based on base
        assert parse_month_generic("Feb", ("2024", "01")) == 1   # Feb 2024
        assert parse_month_generic("Jan", ("2024", "01")) == 0   # Jan 2024
        assert parse_month_generic("Dec", ("2024", "01")) == 11  # Dec 2024
    
    def test_reverse_format(self):
        """Test parsing of reverse format: 'Feb-2025', 'Aug-2024'."""
        assert parse_month_generic("Feb-2025", ("2024", "01")) == 13  # Feb 2025
        assert parse_month_generic("Aug-2024", ("2024", "01")) == 7   # Aug 2024
        assert parse_month_generic("Dec-2024", ("2024", "01")) == 11  # Dec 2024
    
    def test_slash_format(self):
        """Test parsing of slash format: '2024/07', '2025/02'."""
        assert parse_month_generic("2024/07", ("2024", "01")) == 6   # July 2024
        assert parse_month_generic("2025/02", ("2024", "01")) == 13  # Feb 2025
        assert parse_month_generic("2024/1", ("2024", "01")) == 0    # Jan 2024 (single digit)
    
    def test_iso_format(self):
        """Test parsing of ISO format: '2024-01', '2025-02'."""
        assert parse_month_generic("2024-01", ("2024", "01")) == 0   # Jan 2024
        assert parse_month_generic("2025-02", ("2024", "01")) == 13  # Feb 2025
        assert parse_month_generic("2024-12", ("2024", "01")) == 11  # Dec 2024
    
    def test_case_insensitive(self):
        """Test case insensitive month name parsing."""
        assert parse_month_generic("FEB", ("2024", "01")) == 1
        assert parse_month_generic("feb", ("2024", "01")) == 1
        assert parse_month_generic("Feb", ("2024", "01")) == 1
        assert parse_month_generic("FEBRUARY", ("2024", "01")) == 1
    
    def test_full_month_names(self):
        """Test full month name parsing."""
        assert parse_month_generic("January", ("2024", "01")) == 0
        assert parse_month_generic("February", ("2024", "01")) == 1
        assert parse_month_generic("March", ("2024", "01")) == 2
        assert parse_month_generic("December", ("2024", "01")) == 11
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        assert parse_month_generic("", ("2024", "01")) is None
        assert parse_month_generic("invalid", ("2024", "01")) is None
        assert parse_month_generic("2024-13", ("2024", "01")) is None  # Invalid month
        assert parse_month_generic("2024-00", ("2024", "01")) is None  # Invalid month
        assert parse_month_generic("13/2024", ("2024", "01")) is None  # Invalid format
    
    def test_different_base_dates(self):
        """Test parsing with different base dates."""
        # Base: 2025-06 (June 2025)
        assert parse_month_generic("2025-06", ("2025", "06")) == 0   # Same as base
        assert parse_month_generic("2025-07", ("2025", "06")) == 1   # One month after
        assert parse_month_generic("2025-05", ("2025", "06")) == -1  # One month before
        assert parse_month_generic("2024-06", ("2025", "06")) == -12 # One year before
    
    def test_arabic_digits_in_months(self):
        """Test handling of Arabic digits in month strings."""
        # This should be handled by normalize_txt in the parser
        assert parse_month_generic("٢٠٢٤-٠١", ("2024", "01")) == 0  # Arabic digits for 2024-01
        assert parse_month_generic("٢٠٢٥-٠٢", ("2024", "01")) == 13 # Arabic digits for 2025-02
    
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        assert parse_month_generic("  2025- Feb  ", ("2024", "01")) == 13
        assert parse_month_generic(" Feb ", ("2024", "01")) == 1
        assert parse_month_generic("\t2024/07\n", ("2024", "01")) == 6
    
    def test_fallback_function(self):
        """Test the fallback function with legacy mapping."""
        # Should work with both new parser and legacy mapping
        assert parse_month_with_fallback("2025- Feb", ("2024", "01")) == 13
        assert parse_month_with_fallback("Jan", ("2024", "01")) == 0
        
        # Legacy format that might not be caught by generic parser
        assert parse_month_with_fallback("June", ("2024", "01")) == 5
        assert parse_month_with_fallback("July", ("2024", "01")) == 6
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Non-string inputs
        assert parse_month_generic(None, ("2024", "01")) is None
        assert parse_month_generic(123, ("2024", "01")) is None
        
        # Very long strings
        long_string = "a" * 1000
        assert parse_month_generic(long_string, ("2024", "01")) is None
        
        # Special characters
        assert parse_month_generic("Feb@2024", ("2024", "01")) is None
        assert parse_month_generic("2024#02", ("2024", "01")) is None
    
    def test_year_boundaries(self):
        """Test parsing across year boundaries."""
        # Test with base in middle of year
        base = ("2024", "06")  # June 2024
        
        # Previous months in same year
        assert parse_month_generic("2024-01", base) == -5  # Jan 2024
        assert parse_month_generic("2024-05", base) == -1  # May 2024
        
        # Future months in same year  
        assert parse_month_generic("2024-07", base) == 1   # July 2024
        assert parse_month_generic("2024-12", base) == 6   # Dec 2024
        
        # Next year
        assert parse_month_generic("2025-01", base) == 7   # Jan 2025
        assert parse_month_generic("2025-06", base) == 12  # June 2025


if __name__ == "__main__":
    pytest.main([__file__]) 