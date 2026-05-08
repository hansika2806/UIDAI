import base64
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "datahack.ipynb"

TABLE_CELLS = {
    "state_month_ratios_sample": 65,
    "activity_correlation_matrix": 66,
    "state_stability_sample": 67,
    "lag_features_sample": 69,
    "state_lifecycle_sample": 92,
    "lifecycle_summary_stats": 94,
    "indicator_correlation_matrix": 107,
    "policy_table_sample": 139,
}

IMAGE_SECTIONS = [
    (73, "Governance Quadrant", "AMI vs UPI quadrant with policy interpretation"),
    (74, "VSI Stress Diagnostic", "Operational fragility using maturity and volatility"),
    (75, "Rank Persistence", "Enrollment rank versus update rank persistence"),
    (76, "Stress vs Scale", "Volatility versus overall Aadhaar activity"),
    (77, "Enrollment Distribution", "State-level univariate enrollment distribution"),
    (78, "Demographic Update Distribution", "State-level univariate demographic distribution"),
    (79, "Biometric Update Distribution", "State-level univariate biometric distribution"),
    (80, "Pareto Activity", "Concentration of total Aadhaar activity"),
    (81, "VSI Distribution", "Distribution of volatility across states"),
    (82, "Enrollment vs Demographic", "Bivariate state relationship"),
    (83, "Enrollment vs Biometric", "Bivariate state relationship"),
    (84, "Demographic vs Biometric", "Bivariate state relationship"),
    (85, "Pressure vs Volatility", "Maintenance pressure and volatility"),
    (86, "Maturity vs Stability", "Maturity against operational stability"),
    (87, "State Correlation Heatmap", "Cross-metric state-level correlations"),
    (88, "Lifecycle Stress Maps", "Trivariate governance and lifecycle views"),
    (93, "Lifecycle Pattern", "Enrollment age share versus adult update share"),
    (95, "Lifecycle Pattern Detailed", "Lifecycle view with stronger labeling"),
    (108, "Indicator Heatmap", "Independence and complementarity of AMI, UPI, VSI, TPS"),
    (112, "Stress Scale Diagnostic", "Scale-adjusted operational stress"),
    (113, "Lifecycle Trivariate", "Lifecycle transition with activity scale"),
    (133, "Pressure-Maturity Anomaly", "Mismatch anomaly highlighting"),
    (134, "Operational Risk Map", "Scale plus volatility risk zone"),
    (136, "Indicator Heatmap Duplicate", "Notebook follow-up heatmap"),
    (141, "Operational Risk Map Duplicate", "Notebook follow-up risk chart"),
]


def notebook_exists() -> bool:
    return NOTEBOOK_PATH.exists()


def _read_notebook() -> Dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _get_cell_output_text(cell: Dict) -> str:
    parts: List[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            parts.append("".join(output["text"]))
        elif "data" in output and "text/plain" in output["data"]:
            parts.append("".join(output["data"]["text/plain"]))
    return "\n".join(parts)


class _SimpleTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: List[str] = []
        self.current_row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = False
            self.current_row.append("".join(self.current_cell).strip())
        elif self.in_row and tag == "tr":
            self.in_row = False
            if any(cell != "" for cell in self.current_row):
                self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _table_from_html(html_str: str) -> Optional[pd.DataFrame]:
    parser = _SimpleTableParser()
    parser.feed(html_str)
    rows = [row for row in parser.rows if row]
    if not rows:
        return None

    normalized_rows = []
    max_len = max(len(row) for row in rows)
    for row in rows:
        padded = row + [""] * (max_len - len(row))
        normalized_rows.append(padded)

    if len(normalized_rows) == 1:
        return pd.DataFrame(normalized_rows)

    header = normalized_rows[0]
    body = normalized_rows[1:]
    return pd.DataFrame(body, columns=header)


def _get_cell_html_table(cell: Dict) -> Optional[pd.DataFrame]:
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        html = data.get("text/html")
        if not html:
            continue
        html_str = "".join(html)
        table = _table_from_html(html_str)
        if table is not None:
            return table
    return None


def _extract_metric(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return cast(match.group(1))


def _extract_image_bytes(cell: Dict) -> Optional[bytes]:
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        image_b64 = data.get("image/png")
        if image_b64:
            return base64.b64decode("".join(image_b64))
    return None


def load_notebook_artifacts() -> Dict:
    notebook = _read_notebook()
    cells = notebook["cells"]

    tables: Dict[str, pd.DataFrame] = {}
    for name, idx in TABLE_CELLS.items():
        table = _get_cell_html_table(cells[idx])
        if table is not None:
            tables[name] = table

    texts = {idx: _get_cell_output_text(cells[idx]) for idx in [71, 103, 107, 120, 124, 126, 127, 129, 130, 131, 139]}

    images = []
    for idx, title, caption in IMAGE_SECTIONS:
        image_bytes = _extract_image_bytes(cells[idx])
        if image_bytes:
            images.append(
                {
                    "cell_index": idx,
                    "title": title,
                    "caption": caption,
                    "bytes": image_bytes,
                }
            )

    policy_text = texts[120]
    anomalies_text = texts[131]
    summary_text = texts[71]
    restart_text = texts[124]

    category_counts = {}
    for label, count in re.findall(r"([A-Za-z /\\-\u2013]+)\s+(\d+)", policy_text):
        cleaned = label.strip()
        if cleaned and cleaned != "Name: count, dtype":
            category_counts[cleaned] = int(count)

    rho_values = re.findall(r"[-]?\d+\.\d+", _get_cell_output_text(cells[68]))
    sample_state_match = re.search(
        r"Pressure.*?mismatch states:\s+.*?\n\d+\s+([A-Za-z ]+?)\s+([0-9.]+)\s+([0-9.]+)",
        texts[127],
        flags=re.S | re.I,
    )

    summary = {
        "states_aggregated": _extract_metric(r"States aggregated: (\d+)", summary_text, int),
        "state_master_rows": _extract_metric(r"\((\d+),\s*14\)", restart_text, int),
        "state_month_rows": _extract_metric(r"\(\d+,\s*14\)\s*\((\d+),\s*16\)", restart_text, int),
        "anomaly_state_count": _extract_metric(r"Total anomalous states: (\d+)", anomalies_text, int),
        "lifecycle_correlation": _extract_metric(r"Lifecycle correlation .*: ([\-0-9.]+)", _get_cell_output_text(cells[114]), float),
        "rank_rho_enrolment_demographic": float(rho_values[0]) if len(rho_values) >= 1 else None,
        "rank_rho_enrolment_biometric": float(rho_values[1]) if len(rho_values) >= 2 else None,
        "pressure_mismatch_state": sample_state_match.group(1).strip() if sample_state_match else None,
        "pressure_mismatch_upi": float(sample_state_match.group(2)) if sample_state_match else None,
        "pressure_mismatch_ami": float(sample_state_match.group(3)) if sample_state_match else None,
        "policy_category_counts": category_counts,
    }

    anomaly_names = re.findall(r"\n\d+\s+([A-Za-z][A-Za-z And]+?)\s+\d+\s*$", anomalies_text, flags=re.M)

    return {
        "tables": tables,
        "texts": texts,
        "images": images,
        "summary": summary,
        "anomaly_names": anomaly_names,
    }
