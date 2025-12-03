# tests/test_graph_and_search.py

from __future__ import annotations

import pytest

from src.flight_planner import (
    Flight,
    Itinerary,
    build_graph,
    find_earliest_itinerary,
    find_cheapest_itinerary,
    MIN_LAYOVER_MINUTES,
    parse_time,
)


def f(
    origin: str,
    dest: str,
    num: str,
    depart: str,
    arrive: str,
    econ: int,
    biz: int,
    first: int,
) -> Flight:
    """Helper to build flights with HH:MM times."""
    return Flight(
        origin=origin,
        dest=dest,
        flight_number=num,
        depart=parse_time(depart),
        arrive=parse_time(arrive),
        economy=econ,
        business=biz,
        first=first,
    )


def assert_valid_itinerary_times(itin: Itinerary) -> None:
    """Check chronological order + layovers for all legs."""
    assert not itin.is_empty()
    prev_arr = None
    for fl in itin.flights:
        assert fl.arrive > fl.depart
        if prev_arr is None:
            prev_arr = fl.arrive
            continue
        # Enforce layover constraint
        assert fl.depart >= prev_arr + MIN_LAYOVER_MINUTES
        prev_arr = fl.arrive


def test_build_graph_basic():
    flights = [
        f("A", "B", "F1", "08:00", "09:00", 100, 200, 300),
        f("A", "C", "F2", "10:00", "11:00", 100, 200, 300),
        f("B", "C", "F3", "09:30", "10:30", 100, 200, 300),
    ]
    graph = build_graph(flights)

    assert set(graph.keys()) == {"A", "B"}
    assert {fl.dest for fl in graph["A"]} == {"B", "C"}
    assert {fl.dest for fl in graph["B"]} == {"C"}


def test_earliest_itinerary_direct_vs_connecting():
    # Direct is earlier arrival than connect.
    flights = [
        f("A", "B", "F1", "08:00", "10:00", 300, 600, 900),  # direct
        f("A", "C", "F2", "08:00", "09:00", 200, 500, 800),
        # C->B departs late, so arrives later than direct path
        f("C", "B", "F3", "10:30", "11:30", 200, 500, 800),
    ]
    graph = build_graph(flights)
    itin = find_earliest_itinerary(
        graph, start="A", dest="B", earliest_departure=parse_time("07:00")
    )
    assert isinstance(itin, Itinerary)
    assert len(itin.flights) == 1
    assert itin.flights[0].flight_number == "F1"
    assert_valid_itinerary_times(itin)


def test_earliest_itinerary_uses_connection_when_no_direct():
    # Only A->X->B exists, and it must respect layover.
    flights = [
        f("A", "X", "F1", "08:00", "09:00", 150, 300, 600),
        # Make this exactly MIN_LAYOVER after previous arrival.
        f("X", "B", "F2", "10:00", "11:00", 150, 300, 600),
    ]
    # Adjust F2.depart to be arr(F1) + MIN_LAYOVER
    flights[1] = f(
        "X",
        "B",
        "F2",
        "10:00",
        "11:00",
        150,
        300,
        600,
    )
    # Ensure we really have a valid layover gap
    assert (
        flights[1].depart
        >= flights[0].arrive + MIN_LAYOVER_MINUTES
    )

    graph = build_graph(flights)
    itin = find_earliest_itinerary(
        graph, start="A", dest="B", earliest_departure=parse_time("07:00")
    )
    assert isinstance(itin, Itinerary)
    assert len(itin.flights) == 2
    assert itin.origin == "A"
    assert itin.dest == "B"
    assert_valid_itinerary_times(itin)


def test_earliest_itinerary_respects_layover_and_avoids_too_short_connection():
    # A->X->B path has TOO SHORT layover and would otherwise be earlier.
    flights = [
        f("A", "X", "FX1", "08:00", "09:00", 200, 400, 800),
        # This departs only 30 minutes after arrival (too short layover if min=60).
        f("X", "B", "FX2", "09:30", "10:30", 200, 400, 800),
        # Direct but later arrival
        f("A", "B", "FD1", "09:30", "11:30", 200, 400, 800),
    ]
    graph = build_graph(flights)
    itin = find_earliest_itinerary(
        graph, start="A", dest="B", earliest_departure=parse_time("07:00")
    )
    # Valid implementation should NOT use FX1+FX2 since layover is too short.
    # A simple but correct solution is to just take the direct FD1.
    assert isinstance(itin, Itinerary)
    assert len(itin.flights) == 1
    assert itin.flights[0].flight_number == "FD1"
    assert_valid_itinerary_times(itin)


def test_earliest_itinerary_honors_earliest_departure_cutoff():
    flights = [
        f("A", "B", "Fearly", "08:00", "09:00", 200, 400, 800),
        f("A", "B", "Flate", "10:00", "11:00", 200, 400, 800),
    ]
    graph = build_graph(flights)
    # Earliest departure after 09:00 should skip the first flight.
    itin = find_earliest_itinerary(
        graph, start="A", dest="B", earliest_departure=parse_time("09:00")
    )
    assert isinstance(itin, Itinerary)
    assert len(itin.flights) == 1
    assert itin.flights[0].flight_number == "Flate"


def test_earliest_itinerary_deadend_returns_none():
    flights = [
        f("A", "C", "F1", "08:00", "09:00", 100, 200, 300),
        f("C", "D", "F2", "09:00", "11:00", 100, 200, 300),
    ]
    graph = build_graph(flights)
    itin = find_earliest_itinerary(
        graph, start="A", dest="B", earliest_departure=parse_time("07:00")
    )
    assert itin is None


def test_cheapest_itinerary_prefers_cheaper_path_in_economy():
    # Choose between:
    # - Direct expensive economy
    # - Two-leg cheaper economy
    flights = [
        f("A", "B", "Fdirect", "08:00", "10:00", 400, 600, 900),
        f("A", "X", "Fax", "08:00", "09:00", 150, 400, 800),
        f("X", "B", "Fxb", "10:30", "11:30", 150, 400, 800),
    ]
    graph = build_graph(flights)
    itin = find_cheapest_itinerary(
        graph,
        start="A",
        dest="B",
        earliest_departure=parse_time("07:00"),
        cabin="economy",
    )
    assert isinstance(itin, Itinerary)
    # Two-leg path should be cheaper in economy than direct
    assert len(itin.flights) == 2
    assert {fl.flight_number for fl in itin.flights} == {"Fax", "Fxb"}
    # Check prices
    direct_price = flights[0].price_for("economy")
    path_price = itin.total_price("economy")
    assert path_price < direct_price
    assert_valid_itinerary_times(itin)


def test_cheapest_itinerary_can_differ_by_cabin():
    # For economy, the 2-leg path is cheapest.
    # For business, the direct path is cheaper.
    flights = [
        f("A", "B", "Fdirect", "08:00", "10:00", 400, 500, 900),
        f("A", "X", "Fax", "08:00", "09:00", 150, 400, 800),
        f("X", "B", "Fxb", "10:30", "11:30", 150, 400, 800),
    ]
    graph = build_graph(flights)

    econ_itin = find_cheapest_itinerary(
        graph, "A", "B", parse_time("07:00"), cabin="economy"
    )
    biz_itin = find_cheapest_itinerary(
        graph, "A", "B", parse_time("07:00"), cabin="business"
    )

    assert isinstance(econ_itin, Itinerary)
    assert isinstance(biz_itin, Itinerary)

    # Economy should pick two-leg path.
    assert len(econ_itin.flights) == 2

    # Business should prefer direct path (cheaper business price).
    assert len(biz_itin.flights) == 1
    assert biz_itin.flights[0].flight_number == "Fdirect"


def test_cheapest_itinerary_no_route_returns_none():
    flights = [
        f("A", "C", "F1", "08:00", "09:00", 100, 200, 300),
        f("C", "D", "F2", "09:30", "11:00", 100, 200, 300),
    ]
    graph = build_graph(flights)
    itin = find_cheapest_itinerary(
        graph, "A", "B", parse_time("07:00"), cabin="economy"
    )
    assert itin is None


def test_extremely_long_trip_multiple_legs():
    # A long chain that is the only route from A to E.
    flights = [
        f("A", "B", "F1", "06:00", "07:00", 100, 200, 300),
        f("B", "C", "F2", "08:30", "09:30", 100, 200, 300),
        f("C", "D", "F3", "11:00", "12:00", 100, 200, 300),
        f("D", "E", "F4", "14:00", "15:00", 100, 200, 300),
    ]
    graph = build_graph(flights)
    itin = find_earliest_itinerary(
        graph, "A", "E", parse_time("05:00")
    )
    assert isinstance(itin, Itinerary)
    assert len(itin.flights) == 4
    assert itin.origin == "A"
    assert itin.dest == "E"
    assert_valid_itinerary_times(itin)
