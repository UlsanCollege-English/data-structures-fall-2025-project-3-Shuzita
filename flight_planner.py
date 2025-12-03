from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

MIN_LAYOVER_MINUTES: int = 60
Cabin = Literal["economy", "business", "first"]


@dataclass(frozen=True)
class Flight:
    origin: str
    dest: str
    flight_number: str
    depart: int
    arrive: int
    economy: int
    business: int
    first: int

    def price_for(self, cabin: Cabin) -> int:
        if cabin == "economy":
            return self.economy
        elif cabin == "business":
            return self.business
        elif cabin == "first":
            return self.first
        else:
            raise ValueError("Invalid cabin")


@dataclass
class Itinerary:
    flights: List[Flight]

    def is_empty(self) -> bool:
        return not self.flights

    @property
    def origin(self) -> Optional[str]:
        return self.flights[0].origin if self.flights else None

    @property
    def dest(self) -> Optional[str]:
        return self.flights[-1].dest if self.flights else None

    @property
    def depart_time(self) -> Optional[int]:
        return self.flights[0].depart if self.flights else None

    @property
    def arrive_time(self) -> Optional[int]:
        return self.flights[-1].arrive if self.flights else None

    def total_price(self, cabin: Cabin) -> int:
        return sum(f.price_for(cabin) for f in self.flights)

    def num_stops(self) -> int:
        return max(0, len(self.flights) - 1)


Graph = Dict[str, List[Flight]]


def parse_time(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError("Invalid time")
    return hh * 60 + mm


def format_time(minutes: int) -> str:
    return f"{minutes//60:02d}:{minutes%60:02d}"


def parse_flight_line_txt(line: str) -> Optional[Flight]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) != 8:
        raise ValueError("Expected 8 fields")

    o, d, fn, dep, arr, eco, bus, fst = parts
    depart = parse_time(dep)
    arrive = parse_time(arr)

    if arrive <= depart:
        raise ValueError("Arrival must be after departure")

    return Flight(o, d, fn, depart, arrive, int(eco), int(bus), int(fst))


def load_flights_txt(path: str) -> List[Flight]:
    flights = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            try:
                fl = parse_flight_line_txt(line)
                if fl:
                    flights.append(fl)
            except Exception as e:
                raise ValueError(f"{path}:{lineno}: {e}")
    return flights


def load_flights_csv(path: str) -> List[Flight]:
    flights = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            depart = parse_time(row["depart"])
            arrive = parse_time(row["arrive"])
            if arrive <= depart:
                raise ValueError("Arrival must be after departure")
            flights.append(
                Flight(
                    row["origin"], row["dest"], row["flight_number"],
                    depart, arrive,
                    int(row["economy"]), int(row["business"]), int(row["first"])
                )
            )
    return flights


def load_flights(path: str) -> List[Flight]:
    if Path(path).suffix.lower() == ".csv":
        return load_flights_csv(path)
    return load_flights_txt(path)


def build_graph(flights: Iterable[Flight]) -> Graph:
    graph: Graph = {}
    for fl in flights:
        graph.setdefault(fl.origin, []).append(fl)
    return graph


def reconstruct(prev: Dict[str, Flight], dest: str) -> Itinerary:
    route = []
    cur = dest
    while cur in prev:
        fl = prev[cur]
        route.append(fl)
        cur = fl.origin
    route.reverse()
    return Itinerary(route)


def find_earliest_itinerary(graph: Graph, start: str, dest: str, earliest_departure: int) -> Optional[Itinerary]:
    dist = {start: earliest_departure}
    prev: Dict[str, Flight] = {}
    pq = [(earliest_departure, start)]

    while pq:
        cur_time, airport = heapq.heappop(pq)
        if airport == dest:
            return reconstruct(prev, dest)
        if cur_time > dist.get(airport, float("inf")):
            continue

        for fl in graph.get(airport, []):
            min_dep = earliest_departure if airport == start else cur_time + MIN_LAYOVER_MINUTES
            if fl.depart < min_dep:
                continue

            new_time = fl.arrive
            if new_time < dist.get(fl.dest, float("inf")):
                dist[fl.dest] = new_time
                prev[fl.dest] = fl
                heapq.heappush(pq, (new_time, fl.dest))

    return None


def find_cheapest_itinerary(graph: Graph, start: str, dest: str, earliest_departure: int, cabin: Cabin) -> Optional[Itinerary]:
    cost = {start: 0}
    arrival = {start: earliest_departure}
    prev: Dict[str, Flight] = {}
    pq = [(0, start)]

    while pq:
        cur_cost, airport = heapq.heappop(pq)
        if airport == dest:
            return reconstruct(prev, dest)
        if cur_cost > cost.get(airport, float("inf")):
            continue

        for fl in graph.get(airport, []):
            min_dep = earliest_departure if airport == start else arrival[airport] + MIN_LAYOVER_MINUTES
            if fl.depart < min_dep:
                continue

            new_cost = cur_cost + fl.price_for(cabin)
            if new_cost < cost.get(fl.dest, float("inf")):
                cost[fl.dest] = new_cost
                arrival[fl.dest] = fl.arrive
                prev[fl.dest] = fl
                heapq.heappush(pq, (new_cost, fl.dest))

    return None


@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[Cabin]
    itinerary: Optional[Itinerary]
    note: str = ""


def format_comparison_table(origin: str, dest: str, earliest_departure: int, rows: List[ComparisonRow]) -> str:
    out = []
    out.append(f"Comparison for {origin} → {dest} (≥ {format_time(earliest_departure)})")
    out.append(f"{'Mode':20} {'Cabin':7} Dep    Arr    Duration Stops Price")

    for row in rows:
        if not row.itinerary:
            out.append(f"{row.mode:20} {row.cabin or '-':7} N/A    N/A    N/A      N/A   N/A {row.note}")
        else:
            it = row.itinerary
            dep = format_time(it.depart_time)
            arr = format_time(it.arrive_time)
            dur = it.arrive_time - it.depart_time
            dur_str = f"{dur//60}h{dur%60:02d}m"
            stops = it.num_stops()
            price = it.total_price(row.cabin) if row.cabin else "-"
            out.append(f"{row.mode:20} {row.cabin or '-':7} {dep:5} {arr:5} {dur_str:8} {stops:5} {price}")

    return "\n".join(out)


def run_compare(args: argparse.Namespace):
    earliest = parse_time(args.departure_time)
    flights = load_flights(args.flight_file)
    graph = build_graph(flights)

    res1 = find_earliest_itinerary(graph, args.origin, args.dest, earliest)
    res2 = find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "economy")
    res3 = find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "business")
    res4 = find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "first")

    rows = [
        ComparisonRow("Earliest Arrival", None, res1, "(no valid itinerary)" if not res1 else ""),
        ComparisonRow("Cheapest (Eco)", "economy", res2),
        ComparisonRow("Cheapest (Bus)", "business", res3),
        ComparisonRow("Cheapest (First)", "first", res4),
    ]

    print(format_comparison_table(args.origin, args.dest, earliest, rows))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyWise Flight Comparator")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("flight_file")
    cmp.add_argument("origin")
    cmp.add_argument("dest")
    cmp.add_argument("departure_time")
    cmp.set_defaults(func=run_compare)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
