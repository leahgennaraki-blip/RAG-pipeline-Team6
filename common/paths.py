"""Repository paths, anchored to this file so scripts run from any folder."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RESPONSES_CSV = DATA / "Team6_comparative_analysis.csv"
DOCUSCOPE_INPUT = DATA / "docuscope_input"    # Written by 0_preprocessing/prepare_docuscope_input.py
DOCUSCOPE_EXPORT = DATA / "docuscope_export"  # Written by the DocuScope tool (manual step)


def load_saved(output_dir, filename):
    """Read a CSV that an earlier full run saved, for --from-saved."""
    path = Path(output_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run without --from-saved first.")
    return pd.read_csv(path)
