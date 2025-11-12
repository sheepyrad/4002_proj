"""
Run multi-year Spearman correlation analysis
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from correlation_analysis import analyze_multiple_years

if __name__ == '__main__':
    analyze_multiple_years()

