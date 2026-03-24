"""Room-topology helper utilities for Zelda room-graph matching."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

RoomPos = Tuple[int, int]


def build_room_adjacency(rooms: Dict[RoomPos, Any]) -> Dict[RoomPos, List[RoomPos]]:
    """Build adjacency list from reciprocal room door connections."""
    adjacency: Dict[RoomPos, List[RoomPos]] = {pos: [] for pos in rooms}

    for pos, room in rooms.items():
        row, col = pos

        if room.doors.get("N") and (row - 1, col) in rooms:
            if rooms[(row - 1, col)].doors.get("S"):
                adjacency[pos].append((row - 1, col))

        if room.doors.get("S") and (row + 1, col) in rooms:
            if rooms[(row + 1, col)].doors.get("N"):
                adjacency[pos].append((row + 1, col))

        if room.doors.get("W") and (row, col - 1) in rooms:
            if rooms[(row, col - 1)].doors.get("E"):
                adjacency[pos].append((row, col - 1))

        if room.doors.get("E") and (row, col + 1) in rooms:
            if rooms[(row, col + 1)].doors.get("W"):
                adjacency[pos].append((row, col + 1))

    return adjacency


def room_signature(room: Any) -> Tuple[int, int, int, int]:
    """Return compact room door signature as (N,S,W,E) flags."""
    return (
        1 if room.doors.get("N") else 0,
        1 if room.doors.get("S") else 0,
        1 if room.doors.get("W") else 0,
        1 if room.doors.get("E") else 0,
    )


def find_room_at_distance(
    rooms: Dict[RoomPos, Any],
    room_adjacency: Dict[RoomPos, List[RoomPos]],
    start_pos: RoomPos,
    target_distance: int,
) -> Optional[RoomPos]:
    """Find a room near target BFS distance from start, preferring dead-ends."""
    distances = {start_pos: 0}
    queue = deque([start_pos])

    while queue:
        pos = queue.popleft()
        for neighbor in room_adjacency.get(pos, []):
            if neighbor not in distances:
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)

    candidates = []
    for room_pos, dist in distances.items():
        if room_pos == start_pos:
            continue
        door_count = sum(rooms[room_pos].doors.values())
        is_dead_end = door_count == 1
        distance_diff = abs(dist - target_distance)
        score = (distance_diff, 0 if is_dead_end else 1, -dist)
        candidates.append((score, room_pos))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    return None


def find_entrance_room(
    rooms: Dict[RoomPos, Any],
    logger: Optional[Any] = None,
) -> Optional[RoomPos]:
    """Find room that has a door leading outside known room coordinates."""
    for pos, room in rooms.items():
        row, col = pos

        for direction, has_door in room.doors.items():
            if not has_door:
                continue

            if direction == "N":
                target = (row - 1, col)
            elif direction == "S":
                target = (row + 1, col)
            elif direction == "W":
                target = (row, col - 1)
            elif direction == "E":
                target = (row, col + 1)
            else:
                continue

            if target not in rooms:
                if logger is not None:
                    logger.debug(
                        "ENTRANCE_FOUND: Room %s has %s door leading outside (to %s)",
                        pos,
                        direction,
                        target,
                    )
                return pos

    return None


def find_farthest_dead_end(
    rooms: Dict[RoomPos, Any],
    start_pos: RoomPos,
    room_adjacency_fn,
) -> Optional[RoomPos]:
    """Find dead-end farthest from start; fallback to farthest reachable room."""
    adjacency = room_adjacency_fn(rooms)

    distances = {start_pos: 0}
    queue = deque([start_pos])

    while queue:
        pos = queue.popleft()
        for neighbor in adjacency.get(pos, []):
            if neighbor not in distances:
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)

    farthest = None
    max_dist = -1

    for pos, room in rooms.items():
        door_count = sum(room.doors.values())
        dist = distances.get(pos, 0)
        if door_count == 1 and dist > max_dist:
            max_dist = dist
            farthest = pos

    if farthest is None:
        for pos, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                farthest = pos

    return farthest
