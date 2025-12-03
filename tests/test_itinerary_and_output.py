# tests/test_itinerary_and_output.py

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.flight_planner import (
    Flight,
    Itinerary,
    ComparisonRow,
    format_comparison_table,
    parse_time,
    format_time,
    MIN_LAYOVER_MINUTES,
    build_graph,
    find_earliest_itinerary,
    find_cheapest_itinerary,
    load_flights_txt,
    build_arg_parser,
    main,
)


def make_demo_itinerary() -> Itinerary:
    f1 = Flight(
        origin="ICN",
        dest="NRT",
        flight_number="F1",
        depart=parse_time("08:00"),
        arrive=parse_time("10:00"),
        economy=300,
        business=800,
        first=1500,
    )
    f2 = Flight(
        origin="NRT",
        dest="SFO",
        flight_number="F2",
        depart=parse_time("11:30"),
        arrive=parse_time("19:30"),
        economy=500,
        business=1200,
        first=2000,
    )
    return Itinerary([f1, f2])


def test_itinerary_properties_and_price_and_stops():
    itin = make_demo_itinerary()
    assert itin.origin == "ICN"
    assert itin.dest == "SFO"
    assert format_time(itin.depart_time) == "08:00"
    assert format_time(itin.arrive_time) == "19:30"

    # Two flights -> one stop
    assert itin.num_stops() == 1

    econ_total = itin.total_price("economy")
    biz_total = itin.total_price("business")
    first_total = itin.total_price("first")

    assert econ_total == 300 + 500
    assert biz_total == 800 + 1200
    assert first_total == 1500 + 2000


def test_format_comparison_table_basic():
    itin = make_demo_itinerary()
    rows = [
        ComparisonRow(
            mode="Earliest arrival",
            cabin=None,
            itinerary=itin,
            note="",
        ),
        ComparisonRow(
            mode="Cheapest (Economy)",
            cabin="economy",
            itinerary=None,
            note="no valid itinerary",
        ),
    ]

    table = format_comparison_table(
        origin="ICN",
        dest="SFO",
        earliest_departure=parse_time("07:00"),
        rows=rows,
    )

    # Header should mention key columns.
    assert "Mode" in table
    assert "Cabin" in table
    assert "Dep" in table
    assert "Arr" in table
    assert "Total" in table or "Price" in table

    # First row: should include mode name and some times.
    assert "Earliest arrival" in table
    assert "ICN" in table or "SFO" in table  # route info
    assert "08:00" in table  # depart time
    assert "19:30" in table  # arrive time

    # Second row: should include note about missing itinerary.
    assert "Cheapest (Economy)" in table
    assert "no valid itinerary" in table


def test_end_to_end_small_cli(tmp_path: Path, capsys):
    """
    Small end-to-end test:
    - Create a tiny schedule file.
    - Run the CLI 'compare' command.
    - Check that it prints something sensible (does not crash).
    """
    content = textwrap.dedent(
        """
        # Tiny schedule: ICN -> NRT -> SFO, plus direct ICN->SFO
        ICN NRT FW101 08:00 10:00 300 800 1500
        NRT SFO FW102 11:30 19:30 500 1200 2000
        ICN SFO FW103 09:00 19:00 700 1500 2500
        """
    ).strip()
    path = tmp_path / "tiny_flights.txt"
    path.write_text(content + "\n", encoding="utf-8")

    # Build args as if from command line.
    argv = [
        "compare",
        str(path),
        "ICN",
        "SFO",
        "07:00",
    ]

    # Use the real main() to exercise CLI + everything else.
    main(argv)

    captured = capsys.readouterr().out

    # We don't over-constrain formatting, but we expect at least:
    assert "ICN" in captured
    assert "SFO" in captured
    # We expect at least one of the mode labels
    assert "Cheapest" in captured or "Earliest" in captured
