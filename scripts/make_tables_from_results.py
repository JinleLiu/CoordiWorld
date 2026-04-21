"""Build report tables from real result JSON/CSV files.

This script intentionally does not fabricate benchmark numbers. Without an
input result file it either errors, or emits a clearly marked dry-run table
when --dry-run is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("method", "Method"),
    ("split", "Split"),
    ("epdms", "EPDMS"),
    ("spearman", "Spearman"),
    ("kendall", "Kendall"),
    ("ndcg_at_3", "NDCG@3"),
    ("top1_collision", "Top-1 Collision"),
    ("top1_violation", "Top-1 Violation"),
    ("ece", "ECE"),
    ("brier_score", "Brier Score"),
    ("risk_drop_at_k", "RiskDrop@K"),
    ("entity_recall_at_k", "EntityRecall@K"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create Markdown or CSV tables from real CoordiWorld result JSON/CSV files. "
            "Use --dry-run only for an explicitly marked placeholder table."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Real result JSON/CSV file. Repeat to merge multiple files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional table output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit a clearly marked placeholder table when no real input file is available.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input:
        if not args.dry_run:
            parser.error("at least one --input real result JSON/CSV file is required")
        table = render_table([dry_run_row()], output_format=args.format)
        write_or_print(table, args.output)
        return 0

    if args.dry_run:
        parser.error("--dry-run cannot be combined with --input")

    rows = list(load_result_rows(args.input))
    if not rows:
        raise SystemExit("No result rows found in input files.")
    reject_dry_run_rows(rows)

    table = render_table(rows, output_format=args.format)
    write_or_print(table, args.output)
    return 0


def load_result_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Result file does not exist: {path}")
        if path.suffix.lower() == ".json":
            yield from load_json_rows(path)
        elif path.suffix.lower() == ".csv":
            yield from load_csv_rows(path)
        else:
            raise SystemExit(f"Unsupported result file extension: {path.suffix}")


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if isinstance(data.get("results"), list):
            rows = data["results"]
        elif isinstance(data.get("rows"), list):
            rows = data["rows"]
        else:
            rows = [data]
    else:
        raise SystemExit(f"JSON result must be an object or list: {path}")
    return [require_row_dict(row, path) for row in rows]


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def require_row_dict(row: Any, path: Path) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SystemExit(f"Result rows must be objects in {path}")
    return row


def reject_dry_run_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if parse_bool(row.get("dry_run")) or parse_bool(row.get("benchmark_result")) is False:
            raise SystemExit(
                "Input contains a dry-run or non-benchmark row at index "
                f"{index}. Refusing to create a real result table."
            )


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def render_table(rows: list[dict[str, Any]], *, output_format: str) -> str:
    if output_format == "csv":
        return render_csv(rows)
    if output_format == "markdown":
        return render_markdown(rows)
    raise ValueError(f"Unsupported output format: {output_format}")


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [label for _, label in TABLE_COLUMNS]
    separator = ["---" for _ in TABLE_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        cells = [format_cell(row.get(key)) for key, _ in TABLE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, Any]]) -> str:
    output_rows = [[label for _, label in TABLE_COLUMNS]]
    output_rows.extend([[format_cell(row.get(key)) for key, _ in TABLE_COLUMNS] for row in rows])
    return "\n".join(",".join(escape_csv_cell(cell) for cell in row) for row in output_rows) + "\n"


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def escape_csv_cell(value: str) -> str:
    if any(char in value for char in {",", '"', "\n"}):
        return '"' + value.replace('"', '""') + '"'
    return value


def dry_run_row() -> dict[str, Any]:
    return {
        "method": "DRY-RUN PLACEHOLDER - NOT A BENCHMARK RESULT",
        "split": "synthetic",
        "dry_run": True,
        "benchmark_result": False,
    }


def write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
