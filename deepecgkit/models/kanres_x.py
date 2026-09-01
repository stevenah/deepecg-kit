"""Backwards-compatible re-exports.

This module historically defined KanResWideX (despite its name); the class now
lives in kanres_wide_x.py, matching its name. Import from
``deepecgkit.models`` or ``deepecgkit.models.kanres_wide_x`` in new code.
"""

from deepecgkit.models.kanres_wide_x import ConvBlock, KanResModule, KanResWideX

__all__ = ["ConvBlock", "KanResModule", "KanResWideX"]
