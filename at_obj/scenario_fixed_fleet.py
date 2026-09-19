# -*- coding: utf-8 -*-
"""
UAGMC fixed conserved fleet extension.

Only changes:
- disable automatic replacement spawning during the episode;
- keep a fixed aircraft ID set;
- service aircraft physically return from the common arrival vertiport
  to their own home departure vertiport by empty reposition.

Everything else is inherited from the public UAGMC Scenario.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from at_obj.scenario import Scenario
from at_obj.evtol.vehicle_state import VehicleState


class ConservedFleetScenario(Scenario):
    def __init__(
        self,
        *args,
        fleet_size: Optional[int] = None,
        fleet_departure_vertiports: Optional[Sequence[int | str]] = None,
        fleet_return_vertiport: int | str = 2,
        fleet_assertions: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if fleet_departure_vertiports is None:
            fleet_departure_vertiports = (0, 1)

        self.fixed_fleet_departures: List[str] = [
            str(v) for v in fleet_departure_vertiports
        ]
        self.fixed_fleet_return_vertiport = str(fleet_return_vertiport)
        self.fixed_fleet_assertions = bool(fleet_assertions)

        if not self.fixed_fleet_departures:
            raise ValueError("fleet_departure_vertiports cannot be empty")

        for vid in self.fixed_fleet_departures:
            if vid not in self.vertiports.vertiport_list:
                raise ValueError(f"unknown departure vertiport: {vid}")

        if self.fixed_fleet_return_vertiport not in self.vertiports.vertiport_list:
            raise ValueError(
                f"unknown return vertiport: {self.fixed_fleet_return_vertiport}"
            )

        if fleet_size is None:
            fleet_size = sum(
                int(self.vertiports.vertiport_evtol_capacity.get(vid, 0))
                for vid in self.fixed_fleet_departures
            )

        self.fixed_fleet_size = int(fleet_size)
        if self.fixed_fleet_size <= 0:
            raise ValueError("fleet_size must be positive")

        self._fixed_home: Dict[str, str] = {}
        self._fixed_fleet_initialized = False
        self.fixed_fleet_stats = {}

    def reset(self, seed=None):
        self._fixed_home = {}
        self._fixed_fleet_initialized = False
        self.fixed_fleet_stats = {
            "spawned_at_reset": 0,
            "service_arrivals": 0,
            "reposition_departures": 0,
            "reposition_arrivals": 0,
            "fleet_assertions": 0,
        }
        state = super().reset(seed=seed)
        self._assert_fixed_fleet()
        return state

    def _fleet_weights(self) -> Dict[str, int]:
        w = {
            vid: max(
                0,
                int(self.vertiports.vertiport_evtol_capacity.get(vid, 0)),
            )
            for vid in self.fixed_fleet_departures
        }
        if sum(w.values()) <= 0:
            w = {vid: 1 for vid in self.fixed_fleet_departures}
        return w

    @staticmethod
    def _largest_remainder_allocation(total: int, weights: Dict[str, int]):
        vids = sorted(weights, key=str)
        s = float(sum(weights.values()))
        raw = {v: total * weights[v] / s for v in vids}
        alloc = {v: int(math.floor(raw[v])) for v in vids}
        rem = total - sum(alloc.values())
        order = sorted(
            vids,
            key=lambda v: (-(raw[v] - alloc[v]), str(v)),
        )
        for i in range(rem):
            alloc[order[i % len(order)]] += 1
        return alloc

    def get_fixed_fleet_allocation(self):
        return self._largest_remainder_allocation(
            self.fixed_fleet_size,
            self._fleet_weights(),
        )

    def _initialize_fixed_fleet(self):
        if self._fixed_fleet_initialized:
            return

        if self._all_evtols or self.evtols.evtols:
            raise RuntimeError(
                "fixed-fleet init expected empty aircraft registries after reset"
            )

        alloc = self.get_fixed_fleet_allocation()
        next_id = 0

        for home in sorted(alloc, key=str):
            for _ in range(alloc[home]):
                eid = f"fixed_{next_id}"
                next_id += 1

                # Reuse the official spawn path so model/spec/battery/charging
                # initialization stays the same.
                self._spawn_landing_evtol(eid, home)
                self._fixed_home[eid] = home

        self.eVTOL_last_id = next_id
        self._fixed_fleet_initialized = True
        self.fixed_fleet_stats["spawned_at_reset"] = next_id

        if next_id != self.fixed_fleet_size:
            raise RuntimeError(
                f"fleet init mismatch: {next_id} != {self.fixed_fleet_size}"
            )

    def _distance_between_vertiports(self, a: str, b: str) -> float:
        p1 = self.vertiports.vertiport_list[str(a)].vertiport_position
        p2 = self.vertiports.vertiport_list[str(b)].vertiport_position
        return math.sqrt(
            (float(p2[0]) - float(p1[0])) ** 2
            + (float(p2[1]) - float(p1[1])) ** 2
        )

    def _start_empty_reposition(self, evtol, origin: str, dest: str) -> bool:
        if evtol.state != VehicleState.IDLE or evtol.passenger_ids:
            return False

        distance = self._distance_between_vertiports(origin, dest)
        energy = distance * evtol.spec.energy_consumption_kwh_per_km

        # Same minimum energy test used by official service dispatch.
        if evtol.battery_kwh < energy:
            evtol.state = VehicleState.CHARGING
            return False

        evtol.target_vertiport_id = str(dest)
        evtol.passenger_ids = []
        evtol.remaining_time = distance / evtol.spec.max_speed * 60.0
        evtol.flight_distance_km = distance
        evtol.planned_energy_kwh = energy
        evtol.battery_kwh -= energy
        evtol.state = VehicleState.FLYING

        local = self.vertiports.evtols_at_vertiport[str(origin)]
        if evtol in local:
            local.remove(evtol)

        self.fixed_fleet_stats["reposition_departures"] += 1
        return True

    def _dispatch_fixed_returns(self):
        arrival_vid = self.fixed_fleet_return_vertiport
        local = list(
            self.vertiports.evtols_at_vertiport.get(arrival_vid, [])
        )

        for evtol in local:
            home = self._fixed_home.get(evtol.id)
            if home is None:
                raise RuntimeError(f"missing home mapping for {evtol.id}")

            if (
                str(evtol.current_vertiport_id) == arrival_vid
                and home != arrival_vid
                and evtol.state == VehicleState.IDLE
                and not evtol.passenger_ids
            ):
                self._start_empty_reposition(evtol, arrival_vid, home)

    def _maintain_evtol_capacity(self):
        # Called by official reset and once per official Scenario.step.
        if not self._fixed_fleet_initialized:
            self._initialize_fixed_fleet()
        else:
            self._dispatch_fixed_returns()
        self._assert_fixed_fleet()

    def _register_arrived_evtol(self, evtol):
        vid = str(evtol.current_vertiport_id)
        local = self.vertiports.evtols_at_vertiport[vid]
        if evtol not in local:
            local.append(evtol)

        # Preserve _finish_flight's charging semantics instead of allowing
        # a just-landed aircraft to disappear from the local registry.
        if evtol.battery_kwh >= evtol.spec.battery_capacity_kwh:
            evtol.state = VehicleState.IDLE
        else:
            evtol.state = VehicleState.CHARGING

    def _handle_evtol_arrivals(self):
        arrived_events = []

        for evtol in self._all_evtols.values():
            if not getattr(evtol, "just_arrived", False):
                continue

            eid = evtol.id
            dest = str(evtol.current_vertiport_id)
            passenger_ids = list(evtol.passenger_ids)

            if passenger_ids:
                # Same passenger state semantics as official Scenario.
                for pid in passenger_ids:
                    person = self.persons.persons[pid]
                    person.state = "finished"
                    person.sub_state = "arrived"

                arrived_events.append((eid, passenger_ids, dest))
                self.fixed_fleet_stats["service_arrivals"] += 1
            else:
                # Empty aircraft => return-to-home reposition arrival.
                self.fixed_fleet_stats["reposition_arrivals"] += 1

            evtol.passenger_ids.clear()
            evtol.just_arrived = False
            self._register_arrived_evtol(evtol)

        return arrived_events

    def _assert_fixed_fleet(self):
        if not self.fixed_fleet_assertions or not self._fixed_fleet_initialized:
            return

        self.fixed_fleet_stats["fleet_assertions"] += 1

        all_ids = set(self._all_evtols)
        builder_ids = set(self.evtols.evtols)

        if len(all_ids) != self.fixed_fleet_size:
            raise AssertionError(
                f"fleet conservation failed: {len(all_ids)} "
                f"!= {self.fixed_fleet_size}"
            )
        if builder_ids != all_ids:
            raise AssertionError(
                "Scenario._all_evtols and eVTOLBuilder.evtols differ"
            )

        membership = {eid: 0 for eid in all_ids}
        for evtols in self.vertiports.evtols_at_vertiport.values():
            for evtol in evtols:
                if evtol.id not in membership:
                    raise AssertionError(
                        f"non-fixed aircraft in local registry: {evtol.id}"
                    )
                membership[evtol.id] += 1

        for evtol in self._all_evtols.values():
            n = membership[evtol.id]
            if evtol.state == VehicleState.FLYING:
                if n != 0:
                    raise AssertionError(
                        f"flying aircraft {evtol.id} has local membership={n}"
                    )
            elif n != 1:
                raise AssertionError(
                    f"non-flying aircraft {evtol.id} has local membership={n}"
                )

            if evtol.id not in self._fixed_home:
                raise AssertionError(
                    f"aircraft {evtol.id} missing home mapping"
                )

    def get_fixed_fleet_diagnostics(self):
        states = {}
        locations = {}
        for e in self._all_evtols.values():
            states[e.state.name] = states.get(e.state.name, 0) + 1
            loc = str(e.current_vertiport_id)
            locations[loc] = locations.get(loc, 0) + 1

        return {
            "fleet_size": self.fixed_fleet_size,
            "initial_allocation": self.get_fixed_fleet_allocation(),
            "state_counts": states,
            "current_location_counts": locations,
            **self.fixed_fleet_stats,
        }
