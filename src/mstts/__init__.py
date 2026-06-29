"""
Main module for mstts package.

MSTTS - Multi-Class Stratified Train Test Spliter
"""

from mstts.mstts import get_row_selection, train_test_split

__all__ = ["get_row_selection", "train_test_split"]
