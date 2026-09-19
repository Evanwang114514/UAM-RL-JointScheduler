''' 
Author: PangAY
Date: 2023-12-08 21:31:06
LastEditTime: 2024-01-22 13:15:24
LastEditors: pangay 1623253042@qq.com
'''
import random
import csv
from typing import Dict, List, Any, Optional
from loguru import logger

from at_obj.person.person import PersonInfo
from at_obj.map.map import Map


class PersonBuilder:
    """
    Passenger manager for UAM scenario.
    Responsibilities:
    - spawn passengers dynamically based on time or from a file
    - update passenger states
    - provide list of passengers needing decision
    """

    def __init__(
        self,
        map: Any = Map(),
        max_spawn_per_step: int = 4,
        spawn_prob: float = 0.5,
        spawn_file: Optional[str] = None  # CSV file path
    ) -> None:
        self.map = map
        self.max_spawn_per_step = max_spawn_per_step
        self.spawn_prob = spawn_prob
        self.spawn_file = spawn_file

        self.time = 0
        self.persons: Dict[str, PersonInfo] = {}
        self.file_data: Dict[int, List[Dict]] = {}

        if spawn_file is not None:
            self._load_spawn_file(spawn_file)

    # =========================
    # Load passenger spawn info from CSV
    # CSV format: time,origin_x,origin_y,dest_x,dest_y
    # =========================
    def _load_spawn_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = int(row['time'])
                if t not in self.file_data:
                    self.file_data[t] = []
                self.file_data[t].append({
                    'origin': [int(row['origin_x']), int(row['origin_y'])],
                    'destination': [int(row['dest_x']), int(row['dest_y'])]
                })

    # =========================
    # Reset environment
    # =========================
    def reset(self):
        self.time = 0
        self.persons.clear()

    # =========================
    # Spawn passengers
    # =========================
    def spawn(self, time: int) -> List[str]:
        """
        Spawn passengers at the given time.
        Returns list of person_id that need decision.
        """
        self.time = time
        new_ids = []

        # 文件生成模式
        # if spawn_file is provided，而不是时间，有的时间可能没有乘客，需要 double check
        if self.spawn_file != None:
            if time in self.file_data: 
                for i, info in enumerate(self.file_data[time]):
                    pid = f"{time}_{i}" # 认为设定 id 
                    person = self._create_person(pid, origin=info['origin'], destination=info['destination'])
                    self.persons[pid] = person
                    new_ids.append(pid)
                return new_ids
            else:
                return new_ids

        # 随机生成模式
        num_to_spawn = 0
        for _ in range(self.max_spawn_per_step):
            if random.random() < self.spawn_prob:
                num_to_spawn += 1

        for i in range(num_to_spawn):
            pid = f"{time}_{i}"
            person = self._create_person(pid)
            self.persons[pid] = person
            new_ids.append(pid)

        return new_ids

    # =========================
    # Create individual passenger
    # =========================
    def _create_person(
        self,
        person_id: str,
        origin: Optional[List[int]] = None,
        destination: Optional[List[int]] = None
    ) -> PersonInfo:
        if origin is None:
            origin = [
                random.randint(int(0.1 * self.map.map_len), int(0.2 * self.map.map_len)),
                random.randint(int(0.1 * self.map.map_len), int(0.2 * self.map.map_len))
            ]
        if destination is None:
            destination = [
                random.randint(int(0.8 * self.map.map_len), int(0.9 * self.map.map_len)),
                random.randint(int(0.8 * self.map.map_len), int(0.9 * self.map.map_len))
            ]

        person = PersonInfo(
            person_id,
            spawn_time=self.time,
            origin_position=origin,
            destination_position=destination
        )

        person.state = "new"
        return person

    # =========================
    # Update all passengers
    # =========================
    def update_objects_state(self, time: int) -> List[str]:
        self.time = time
        finished = []

        for pid, person in self.persons.items():
            if person.state not in ("removed",):
                person.update(time)

            if person.state == "d":
                finished.append(pid)

        return finished

    # =========================
    # Query passengers needing decision
    # =========================
    def get_waiting_decisions(self) -> List[str]:
        return [pid for pid, p in self.persons.items() if p.state == "new"]

    # =========================
    # Get current states for all passengers
    # =========================
    def get_state(self) -> Dict[str, Any]:
        return {pid: p.get_observation() for pid, p in self.persons.items()}
