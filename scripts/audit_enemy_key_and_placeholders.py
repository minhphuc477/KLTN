"""Audit enemy/key distribution and placeholder/empty logic in src/."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.definitions import SEMANTIC_PALETTE, parse_node_label_tokens
from src.evaluation.benchmark_suite import load_vglc_reference_graphs, load_vglc_reference_rooms


def _summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "nonzero_rate": 0.0,
        }
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "nonzero_rate": float(np.mean(arr > 0)),
    }


def audit_enemy_key_distribution(data_root: Path, max_rooms: int = 20000) -> Dict[str, Any]:
    rooms = load_vglc_reference_rooms(data_root=data_root, max_rooms=max_rooms)
    graphs = load_vglc_reference_graphs(data_root=data_root, limit=None)

    enemy_id = int(SEMANTIC_PALETTE["ENEMY"])
    key_ids = {
        int(SEMANTIC_PALETTE["KEY_SMALL"]),
        int(SEMANTIC_PALETTE["KEY_BOSS"]),
        int(SEMANTIC_PALETTE["KEY_ITEM"]),
    }

    room_enemy_tiles: List[int] = []
    room_key_tiles: List[int] = []
    for room in rooms:
        grid = np.asarray(room, dtype=np.int32)
        room_enemy_tiles.append(int(np.sum(grid == enemy_id)))
        room_key_tiles.append(int(np.sum(np.isin(grid, list(key_ids)))))

    keys_per_graph: List[int] = []
    enemies_per_graph: List[int] = []
    nodes_per_graph: List[int] = []
    key_nodes_total = 0
    enemy_nodes_total = 0
    node_total = 0
    composite_total = 0

    for g in graphs:
        kc = 0
        ec = 0
        for _, attrs in g.nodes(data=True):
            node_total += 1
            tokens = set(parse_node_label_tokens(str(attrs.get("label", "") or "")))
            if "k" in tokens or "key" in tokens:
                kc += 1
                key_nodes_total += 1
            if "e" in tokens or "enemy" in tokens or "b" in tokens or "boss" in tokens:
                ec += 1
                enemy_nodes_total += 1
            if len(tokens) > 1:
                composite_total += 1
        keys_per_graph.append(int(kc))
        enemies_per_graph.append(int(ec))
        nodes_per_graph.append(int(g.number_of_nodes()))

    return {
        "data_root": str(data_root),
        "num_rooms": int(len(rooms)),
        "num_graphs": int(len(graphs)),
        "room_enemy_tile_stats": _summary(room_enemy_tiles),
        "room_key_tile_stats": _summary(room_key_tiles),
        "keys_per_graph_stats": _summary(keys_per_graph),
        "enemies_per_graph_stats": _summary(enemies_per_graph),
        "nodes_per_graph_stats": _summary(nodes_per_graph),
        "graph_label_rates": {
            "enemy_node_rate": float(enemy_nodes_total / max(1, node_total)),
            "key_node_rate": float(key_nodes_total / max(1, node_total)),
            "composite_label_rate": float(composite_total / max(1, node_total)),
        },
    }


@dataclass
class PlaceholderHit:
    file: str
    line: int
    kind: str
    symbol: str
    classification: str


def _function_name(stack: List[str], node_name: str) -> str:
    if not stack:
        return node_name
    return ".".join(stack + [node_name])


def _decorator_ids(func: ast.AST) -> List[str]:
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    out: List[str] = []
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name):
            out.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            out.append(dec.attr)
        else:
            out.append(ast.dump(dec, include_attributes=False))
    return out


def _collect_placeholder_hits(py_path: Path) -> List[PlaceholderHit]:
    text = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(py_path))
    except SyntaxError:
        return []

    hits: List[PlaceholderHit] = []
    class_stack: List[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            class_stack.append(node.name)
            for child in node.body:
                walk(child)
            class_stack.pop()
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = set(_decorator_ids(node))
            qname = _function_name(class_stack, node.name)
            class_name = class_stack[-1] if class_stack else ""
            base_like = class_name.lower().startswith("base") or class_name in {
                "FeatureExtractor",
                "ProductionRule",
                "TileSet",
            }
            doc = ast.get_docstring(node) or ""
            override_like = "override" in doc.lower()
            body = list(node.body)
            real_body = body[1:] if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else body
            if len(real_body) == 1 and isinstance(real_body[0], ast.Pass):
                if "abstractmethod" in decs:
                    cls = "expected_abstract"
                elif node.name == "__init__":
                    cls = "noop_constructor"
                elif base_like or override_like:
                    cls = "expected_base_contract"
                else:
                    cls = "potential_empty_logic"
                hits.append(
                    PlaceholderHit(
                        file=str(py_path.as_posix()),
                        line=int(real_body[0].lineno),
                        kind="pass_only_function",
                        symbol=qname,
                        classification=cls,
                    )
                )

            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    exc = child.exc
                    not_impl = False
                    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                        not_impl = True
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                        not_impl = True
                    if not not_impl:
                        continue
                    if "abstractmethod" in decs:
                        cls = "expected_abstract"
                    elif base_like or override_like:
                        cls = "expected_base_contract"
                    else:
                        cls = "potential_empty_logic"
                    hits.append(
                        PlaceholderHit(
                            file=str(py_path.as_posix()),
                            line=int(child.lineno),
                            kind="not_implemented_error",
                            symbol=qname,
                            classification=cls,
                        )
                    )

            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    body2 = list(child.body)
                    if len(body2) == 1 and isinstance(body2[0], ast.Pass):
                        hits.append(
                            PlaceholderHit(
                                file=str(py_path.as_posix()),
                                line=int(body2[0].lineno),
                                kind="except_pass",
                                symbol=qname,
                                classification="exception_guard",
                            )
                        )
            return

        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return hits


def audit_placeholders(src_root: Path) -> Dict[str, Any]:
    py_files = sorted(p for p in src_root.rglob("*.py") if p.is_file())
    all_hits: List[PlaceholderHit] = []
    for py in py_files:
        all_hits.extend(_collect_placeholder_hits(py))

    by_class: Dict[str, int] = {}
    for h in all_hits:
        by_class[h.classification] = int(by_class.get(h.classification, 0) + 1)

    potential = [h for h in all_hits if h.classification == "potential_empty_logic"]
    return {
        "src_root": str(src_root),
        "python_files_scanned": int(len(py_files)),
        "total_hits": int(len(all_hits)),
        "classification_counts": by_class,
        "potential_empty_logic": [asdict(h) for h in potential],
        "all_hits": [asdict(h) for h in all_hits],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit enemy/key distributions and placeholder logic.")
    p.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    p.add_argument("--src-root", type=Path, default=Path("src"))
    p.add_argument("--max-rooms", type=int, default=20000)
    p.add_argument("--output", type=Path, default=Path("results") / "enemy_key_placeholder_audit.json")
    p.add_argument("--pretty", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "enemy_key_distribution": audit_enemy_key_distribution(
            data_root=args.data_root,
            max_rooms=max(1, int(args.max_rooms)),
        ),
        "placeholder_audit": audit_placeholders(src_root=args.src_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
