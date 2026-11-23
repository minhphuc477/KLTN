"""Topology-Equivariant Room Symmetry Operations.

Provides Klein four-group (V4 = {I, flip_h, flip_v, flip_h o flip_v}) dihedral
symmetries for rectangular room grids (16x11) and lockstep transformations for
all paired graph conditioning metadata (boundary constraints, directional topology
channels, neighbor maps, and puzzle stage conditions).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.core.definitions import (
    ROOM_HEIGHT,
    ROOM_TOPOLOGY_CHANNELS,
    ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS,
    ROOM_WIDTH,
)

__all__ = [
    "RandomD4Symmetry",
    "apply_room_symmetry_transform",
    "_remap_symmetry_direction",
    "_remap_symmetry_anchor_name",
    "_flip_spatial_tensor_or_array",
    "_swap_topology_directional_channels",
    "_flip_room_coord_pair",
    "_transform_puzzle_stage_condition",
]


class RandomD4Symmetry:
    """Apply a random shape-preserving dihedral transform to a 2D tile grid.

    Rectangular Zelda rooms are stored as `(ROOM_HEIGHT, ROOM_WIDTH)`. A 90 degree
    rotation would change that to `(ROOM_WIDTH, ROOM_HEIGHT)` and make DataLoader
    stacking fail unless every coupled graph/neighbor tensor is rotated too. For
    rectangular grids, restrict the random rotation to 0 or 180 degrees; square
    grids still receive the full D4 group.
    """

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        height, width = int(grid.shape[-2]), int(grid.shape[-1])
        if height == width:
            k = torch.randint(0, 4, ()).item()  # 0,1,2,3 rotations
        else:
            k = int(torch.randint(0, 2, ()).item()) * 2  # keep rectangular shape
        flip = torch.randint(0, 2, ()).item()  # 0=no flip, 1=flip
        if flip:
            grid = torch.flip(grid, dims=[-1])  # horizontal flip
        grid = torch.rot90(grid, k=k, dims=[-2, -1])
        return grid


_SYMMETRY_FLIP_DIRECTION_V: Dict[str, str] = {"N": "S", "S": "N", "n": "s", "s": "n"}
_SYMMETRY_FLIP_DIRECTION_H: Dict[str, str] = {"E": "W", "W": "E", "e": "w", "w": "e"}


def _remap_symmetry_direction(direction: Any, *, flip_h: bool, flip_v: bool) -> Any:
    """Remap a cardinal direction token under horizontal/vertical room flips."""
    if not isinstance(direction, str):
        return direction
    out = direction
    if flip_v and out in _SYMMETRY_FLIP_DIRECTION_V:
        out = _SYMMETRY_FLIP_DIRECTION_V[out]
    if flip_h and out in _SYMMETRY_FLIP_DIRECTION_H:
        out = _SYMMETRY_FLIP_DIRECTION_H[out]
    return out


def _remap_symmetry_anchor_name(name: Any, *, flip_h: bool, flip_v: bool) -> Any:
    """Remap stage/anchor identifiers such as ``'door:N'`` under room symmetry flips."""
    if not isinstance(name, str):
        return name
    if ":" in name:
        prefix, suffix = name.rsplit(":", 1)
        if suffix.upper() in {"N", "S", "E", "W"}:
            return f"{prefix}:{_remap_symmetry_direction(suffix, flip_h=flip_h, flip_v=flip_v)}"
    return _remap_symmetry_direction(name, flip_h=flip_h, flip_v=flip_v)


def _flip_spatial_tensor_or_array(value: Any, *, flip_h: bool, flip_v: bool) -> Any:
    """Flip trailing ``(H, W)`` spatial axes of a tensor or NumPy array."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        dims: List[int] = []
        if flip_v and value.dim() >= 2:
            dims.append(-2)
        if flip_h and value.dim() >= 1:
            dims.append(-1)
        return torch.flip(value, dims=dims) if dims else value.clone()
    if isinstance(value, np.ndarray):
        axes: List[int] = []
        if flip_v and value.ndim >= 2:
            axes.append(-2)
        if flip_h and value.ndim >= 1:
            axes.append(-1)
        return np.flip(value, axis=tuple(axes)).copy() if axes else value.copy()
    return value


def _swap_topology_directional_channels(
    topo_map: torch.Tensor,
    *,
    flip_h: bool,
    flip_v: bool,
) -> torch.Tensor:
    """Swap directional door and gate channels in ``room_topology_map``."""
    out = topo_map.clone()
    if out.dim() < 3:
        return out
    num_channels = int(out.shape[-3])
    swap_pairs: List[Tuple[str, str]] = []
    if flip_v:
        swap_pairs.append(("N", "S"))
    if flip_h:
        swap_pairs.append(("E", "W"))
    for dir_a, dir_b in swap_pairs:
        group_a = ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS.get(dir_a, ())
        group_b = ROOM_TOPOLOGY_DIRECTIONAL_CHANNEL_GROUPS.get(dir_b, ())
        for ch_name_a, ch_name_b in zip(group_a, group_b):
            idx_a = ROOM_TOPOLOGY_CHANNELS.get(ch_name_a)
            idx_b = ROOM_TOPOLOGY_CHANNELS.get(ch_name_b)
            if idx_a is None or idx_b is None:
                continue
            if idx_a >= num_channels or idx_b >= num_channels:
                continue
            tmp = out[..., idx_a, :, :].clone()
            out[..., idx_a, :, :] = out[..., idx_b, :, :]
            out[..., idx_b, :, :] = tmp
    return out


def _flip_room_coord_pair(
    coord: Any,
    *,
    height: int,
    width: int,
    flip_h: bool,
    flip_v: bool,
) -> Any:
    """Transform a ``(row, col)`` room-local coordinate under V4 reflection."""
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return coord
    row, col = coord[0], coord[1]
    if isinstance(row, int) and not isinstance(row, bool):
        new_row: Any = (height - 1 - row) if flip_v else row
    elif isinstance(row, float):
        new_row = (float(height - 1) - row) if flip_v else row
    else:
        new_row = row
    if isinstance(col, int) and not isinstance(col, bool):
        new_col: Any = (width - 1 - col) if flip_h else col
    elif isinstance(col, float):
        new_col = (float(width - 1) - col) if flip_h else col
    else:
        new_col = col
    return [new_row, new_col] if isinstance(coord, list) else (new_row, new_col)


def _transform_puzzle_stage_condition(
    condition: Any,
    *,
    height: int,
    width: int,
    flip_h: bool,
    flip_v: bool,
) -> Any:
    """Transform room-local ``puzzle_stage_condition`` metadata in lockstep."""
    if not isinstance(condition, dict) or not condition:
        return dict(condition) if isinstance(condition, dict) else condition
    out = dict(condition)
    if "controlled_doors" in out and isinstance(out["controlled_doors"], (list, tuple)):
        out["controlled_doors"] = [
            _remap_symmetry_anchor_name(d, flip_h=flip_h, flip_v=flip_v)
            for d in out["controlled_doors"]
        ]
    if "canonical_sequence" in out and isinstance(out["canonical_sequence"], (list, tuple)):
        out["canonical_sequence"] = [
            _remap_symmetry_anchor_name(name, flip_h=flip_h, flip_v=flip_v)
            for name in out["canonical_sequence"]
        ]
    if "anchor_sources" in out and isinstance(out["anchor_sources"], dict):
        out["anchor_sources"] = {
            _remap_symmetry_anchor_name(k, flip_h=flip_h, flip_v=flip_v): v
            for k, v in out["anchor_sources"].items()
        }
    if "stage_trace_mask" in out and out["stage_trace_mask"] is not None:
        out["stage_trace_mask"] = _flip_spatial_tensor_or_array(
            out["stage_trace_mask"], flip_h=flip_h, flip_v=flip_v
        )
    if "stage_sequence" in out and isinstance(out["stage_sequence"], (list, tuple)):
        transformed_stages = []
        for stage in out["stage_sequence"]:
            if not isinstance(stage, dict):
                transformed_stages.append(stage)
                continue
            stage_copy = dict(stage)
            if "name" in stage_copy:
                stage_copy["name"] = _remap_symmetry_anchor_name(
                    stage_copy["name"], flip_h=flip_h, flip_v=flip_v
                )
            for coord_key in ("local_anchor", "start", "goal", "anchor", "position", "source", "target", "coord"):
                if coord_key in stage_copy:
                    stage_copy[coord_key] = _flip_room_coord_pair(
                        stage_copy[coord_key], height=height, width=width, flip_h=flip_h, flip_v=flip_v
                    )
            for dir_key in ("direction", "gate_direction", "entry_direction", "exit_direction"):
                if dir_key in stage_copy:
                    stage_copy[dir_key] = _remap_symmetry_direction(
                        stage_copy[dir_key], flip_h=flip_h, flip_v=flip_v
                    )
            transformed_stages.append(stage_copy)
        out["stage_sequence"] = transformed_stages
    for coord_key in ("local_anchor", "start", "goal", "anchor", "position", "source", "target", "coord"):
        if coord_key in out:
            out[coord_key] = _flip_room_coord_pair(
                out[coord_key], height=height, width=width, flip_h=flip_h, flip_v=flip_v
            )
    for dir_key in ("direction", "gate_direction", "entry_direction", "exit_direction"):
        if dir_key in out:
            out[dir_key] = _remap_symmetry_direction(
                out[dir_key], flip_h=flip_h, flip_v=flip_v
            )
    return out


def apply_room_symmetry_transform(
    room_tensor: torch.Tensor,
    graph_dict: Optional[Dict[str, Any]] = None,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Apply a shape-preserving V4 reflection symmetry to a room and its graph metadata.

    Rectangular Zelda rooms (``16 x 11``) support the Klein four-group
    ``V4 = {I, flip_h, flip_v, flip_h o flip_v}`` without changing tensor shapes.
    All room-local spatial masks, directional topology channels (``N <-> S``,
    ``E <-> W``), ``boundary_constraints``, ``neighbor_maps``, and
    ``puzzle_stage_condition`` metadata are transformed in lockstep.

    Returns ``transformed_room`` when ``graph_dict is None``, or
    ``(transformed_room, transformed_graph_dict)`` otherwise.
    """
    out_room = _flip_spatial_tensor_or_array(room_tensor, flip_h=flip_h, flip_v=flip_v)
    if graph_dict is None:
        return out_room

    height = int(out_room.shape[-2]) if out_room.dim() >= 2 else ROOM_HEIGHT
    width = int(out_room.shape[-1]) if out_room.dim() >= 1 else ROOM_WIDTH
    out_graph = dict(graph_dict)

    for mask_key in ("logic_source_mask", "logic_target_mask"):
        if mask_key in out_graph and out_graph[mask_key] is not None:
            out_graph[mask_key] = _flip_spatial_tensor_or_array(
                out_graph[mask_key], flip_h=flip_h, flip_v=flip_v
            )

    topo = out_graph.get("room_topology_map")
    if isinstance(topo, torch.Tensor):
        flipped_topo = _flip_spatial_tensor_or_array(topo, flip_h=flip_h, flip_v=flip_v)
        out_graph["room_topology_map"] = _swap_topology_directional_channels(
            flipped_topo, flip_h=flip_h, flip_v=flip_v
        )
    elif isinstance(topo, np.ndarray):
        topo_t = torch.from_numpy(topo)
        flipped_topo = _flip_spatial_tensor_or_array(topo_t, flip_h=flip_h, flip_v=flip_v)
        out_graph["room_topology_map"] = _swap_topology_directional_channels(
            flipped_topo, flip_h=flip_h, flip_v=flip_v
        ).numpy()

    bc = out_graph.get("boundary_constraints")
    if isinstance(bc, torch.Tensor) and bc.shape[-1] >= 8:
        bc_out = bc.clone()
        if flip_v:
            tmp_ns = bc_out[..., 0:2].clone()
            bc_out[..., 0:2] = bc_out[..., 2:4]
            bc_out[..., 2:4] = tmp_ns
        if flip_h:
            tmp_ew = bc_out[..., 4:6].clone()
            bc_out[..., 4:6] = bc_out[..., 6:8]
            bc_out[..., 6:8] = tmp_ew
        out_graph["boundary_constraints"] = bc_out
    elif isinstance(bc, np.ndarray) and bc.shape[-1] >= 8:
        bc_out = bc.copy()
        if flip_v:
            tmp_ns = bc_out[..., 0:2].copy()
            bc_out[..., 0:2] = bc_out[..., 2:4]
            bc_out[..., 2:4] = tmp_ns
        if flip_h:
            tmp_ew = bc_out[..., 4:6].copy()
            bc_out[..., 4:6] = bc_out[..., 6:8]
            bc_out[..., 6:8] = tmp_ew
        out_graph["boundary_constraints"] = bc_out

    neighbor_maps = out_graph.get("neighbor_maps")
    if isinstance(neighbor_maps, dict):
        transformed_neighbors: Dict[str, Any] = {}
        for direction, nmap in neighbor_maps.items():
            new_dir = _remap_symmetry_direction(direction, flip_h=flip_h, flip_v=flip_v)
            transformed_neighbors[new_dir] = (
                None
                if nmap is None
                else _flip_spatial_tensor_or_array(nmap, flip_h=flip_h, flip_v=flip_v)
            )
        out_graph["neighbor_maps"] = transformed_neighbors

    if "puzzle_stage_condition" in out_graph:
        out_graph["puzzle_stage_condition"] = _transform_puzzle_stage_condition(
            out_graph["puzzle_stage_condition"],
            height=height,
            width=width,
            flip_h=flip_h,
            flip_v=flip_v,
        )

    return out_room, out_graph
