"""
Chat module - refactored for modularity.

This module now re-exports chat functionality from focused modules to maintain backward compatibility
while providing better organization and maintainability.
"""

# Re-export everything from the chat package for backward compatibility
from .chat import *
