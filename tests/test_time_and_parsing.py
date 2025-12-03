# tests/test_time_and_parsing.py

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.flight_planner import (
    Flight,
    parse_time,
    format_time,
    parse_flight_line_txt,
    load_flights_txt,
    load_flights_csv,
    load_flights,
)


def test_parse_time_roundtrip():
    # Roundtrip a few representative minute values.
    mins = [0, 5, 60, 510, 13 * 60 + 7, 23 * 60 + 59]
    for m in mins:
        s = format_time(m)
        back = parse_time(s)
        assert back == m


@pytest.mark.parametrize(
    "bad",
    [
        "25:00",   # hour too high
        "23:60",   # minute too high
        "-1:00",   # negative hour
        "aa:bb",   # not ints
        "123",     # no colon
        "12-30",   # wrong separator
    ],
)
def test_parse_time_invalid_raises(bad: str):
    with pytest.raises(ValueError):
        parse_time(bad)


def test_parse_flight_line_txt_blank_and_comment():
    assert parse_flight_line_txt("   \n") is None
    assert parse_flight_line_txt("# comment line\n") is None


def test_parse_flight_line_txt_valid_line():
    line = "ICN NRT FW999 08:00 10:00 300 800 1500"
    f = parse_flight_line_txt(line)
    assert isinstance(f, Flight)
    assert f.origin == "ICN"
    assert f.dest == "NRT"
    assert f.flight_number == "FW999"
    assert f.economy == 300
    assert f.business == 800
    assert f.first == 1500
    assert f.arrive > f.depart  # same-day check


def test_parse_flight_line_txt_invalid_arrival_before_departure():
    line = "ICN NRT FW999 10:00 09:00 300 800 1500"
    with pytest.raises(ValueError):
        parse_flight_line_txt(line)


def test_load_flights_txt_basic(tmp_path: Path):
    content = textwrap.dedent(
        """
        # Sample schedule
        ICN NRT FW101 08:00 10:00 300 800 1500

        # another flight
        NRT ICN FW102 11:00 13:00 320 820 1520
        """
    ).strip()

    path = tmp_path / "flights.txt"
    path.write_text(content + "\n", encoding="utf-8")

    flights = load_flights_txt(str(path))
    assert len(flights) == 2

    codes = {(f.origin, f.dest) for f in flights}
    assert ("ICN", "NRT") in codes
    assert ("NRT", "ICN") in codes


def test_load_flights_csv_basic(tmp_path: Path):
    content = textwrap.dedent(
        """
        origin,dest,flight_number,depart,arrive,economy,business,first
        ICN,NRT,FW101,08:00,10:00,300,800,1500
        NRT,ICN,FW102,11:00,13:00,320,820,1520
        """
    ).lstrip()

    path = tmp_path / "flights.csv"
    path.write_text(content, encoding="utf-8")

    flights = load_flights_csv(str(path))
    assert len(flights) == 2
    assert flights[0].origin == "ICN"
    assert flights[1].dest == "ICN"


def test_load_flights_dispatch_uses_extension(tmp_path: Path):
    # TXT file
    txt = tmp_path / "flights.txt"
    txt.write_text(
        "ICN NRT FW101 08:00 10:00 300 800 1500\n", encoding="utf-8"
    )

    # CSV file
    csv = tmp_path / "flights.csv"
    csv.write_text(
        "origin,dest,flight_number,depart,arrive,economy,business,first\n"
        "NRT,ICN,FW102,11:00,13:00,320,820,1520\n",
        encoding="utf-8",
    )

    flights_txt = load_flights(str(txt))
    flights_csv = load_flights(str(csv))

    assert len(flights_txt) == 1
    assert len(flights_csv) == 1
    assert flights_txt[0].origin == "ICN"
    assert flights_csv[0].origin == "NRT"
