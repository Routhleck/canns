#!/usr/bin/env python3
"""CLI wrapper for the ASA spatial TDA pipeline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from canns.analyzer.data.asa.spatial_tda import main

if __name__ == "__main__":
    raise SystemExit(main())
