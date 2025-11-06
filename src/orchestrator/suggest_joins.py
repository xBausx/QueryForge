# src/orchestrator/suggest_joins.py
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# We reuse your config loader
from .config_loader import load_all_configs


@dataclass
class Suggestion:
    left_table: str
    left_col: str
    right_table: str
    right_pk: str


NAME_FIELDS = {"name", "business_name", "title", "label", "code"}


def _normalize_join_graph(jg) -> Dict[str, List[str]]:
    # convert defaultdict -> dict, dedupe and sort neighbor lists, drop empties
    return {k: sorted(set(v)) for k, v in jg.items() if v}


def _guess_ref_table(col: str) -> Optional[str]:
    """
    Heuristics to map 'dealer_id' -> 'dealers', 'user_id' -> 'users', etc.
    """
    if not col.endswith("_id"):
        return None
    stem = col[:-3]  # drop '_id'
    if not stem:
        return None
    # simple pluralization heuristic
    if stem.endswith("y"):
        return stem[:-1] + "ies"
    if stem.endswith("s"):
        return stem  # already plural-ish
    return stem + "s"


def _pick_pk(entities: Dict[str, Any], table: str) -> Optional[str]:
    meta = entities.get(table) or {}
    pks = meta.get("pk") or []
    if isinstance(pks, list) and pks:
        return pks[0]
    # fallbacks
    for cand in ("id", f"{table[:-1]}_id", f"{table}_id"):
        if cand:
            return cand
    return None


def _find_namey_fields(fields_map: Dict[str, Dict[str, Any]], table: str) -> List[str]:
    out: List[str] = []
    tcols = fields_map.get(table) or {}
    for col in tcols.keys():
        base = col.lower()
        if base in NAME_FIELDS or any(base.endswith(f"_{nf}") for nf in NAME_FIELDS):
            out.append(col)
    # keep it short & predictable
    return sorted(out)[:5]


def suggest_joins(config_dir: Path) -> Tuple[Dict[str, List[str]], List[Suggestion], Dict[str, Dict[str, Any]]]:
    bundle = load_all_configs(config_dir.resolve())
    entities = bundle.entities or {}
    fields = bundle.fields or {}

    tables: Set[str] = set(entities.keys())
    join_graph: Dict[str, List[str]] = defaultdict(list)
    suggestions: List[Suggestion] = []

    for lt in sorted(tables):
        tcols = fields.get(lt) or {}
        for col in tcols.keys():
            if not col.endswith("_id"):
                continue
            rt = _guess_ref_table(col)
            if not rt or rt not in tables:
                continue
            rpk = _pick_pk(entities, rt)
            if not rpk:
                continue
            # record suggestion
            suggestions.append(Suggestion(left_table=lt, left_col=col, right_table=rt, right_pk=rpk))
            if rt not in join_graph[lt]:
                for lt in sorted(tables):
                    tcols = fields.get(lt) or {}
                    for col in tcols.keys():
                        if not col.endswith("_id"):
                            continue
                        rt = _guess_ref_table(col)
                        if not rt or rt not in tables:
                            continue
                        if rt == lt:
                            continue  # <-- avoid self-joins; reduces noise
                        rpk = _pick_pk(entities, rt)
                        if not rpk:
                            continue
                        suggestions.append(Suggestion(left_table=lt, left_col=col, right_table=rt, right_pk=rpk))
                        if rt not in join_graph[lt]:
                            join_graph[lt].append(rt)


    # also prepare a minimal lookup_projections suggestion: expose "name-like" columns
    lookup_suggest: Dict[str, Dict[str, Any]] = {}
    for s in suggestions:
        namey = _find_namey_fields(fields, s.right_table)
        if not namey:
            continue
        base = lookup_suggest.setdefault(s.left_table, {})
        for nn in namey:
            key = f"{s.right_table}.{nn}"
            base[key] = {
                "join": {
                    "table": s.right_table,
                    "on": f"{s.left_table}.{s.left_col} = {s.right_table}.{s.right_pk}",
                    "type": "left"
                }
            }

    return join_graph, suggestions, lookup_suggest


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest join_graph and lookup_projections from *_id columns.")
    ap.add_argument("--config", default="src/orchestrator/config", help="Config dir (default: src/orchestrator/config)")
    ap.add_argument("--write", action="store_true", help="Write join_graph.yaml / lookup_projections.yaml")
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        print(f"Config directory not found: {cfg}", file=sys.stderr)
        sys.exit(2)

    join_graph, suggestions, lookup_suggest = suggest_joins(cfg)

    jg_out = _normalize_join_graph(join_graph)

    print("# --- Suggested join_graph.yaml ---")
    print(yaml.safe_dump(jg_out, sort_keys=True, default_flow_style=False).rstrip())

    print("\n# --- Suggested lookup_projections.yaml (add as needed) ---")
    print(yaml.safe_dump(lookup_suggest, sort_keys=True, default_flow_style=False).rstrip())

    if args.write:
        jg_path = cfg / "join_graph.yaml"
        lp_path = cfg / "lookup_projections.yaml"

        # merge with existing files if they already have content
        def _merge_yaml(existing_path: Path, new_obj: Dict[str, Any]) -> Dict[str, Any]:
            if existing_path.exists() and existing_path.stat().st_size > 0:
                try:
                    with existing_path.open("r", encoding="utf-8") as fh:
                        cur = yaml.safe_load(fh) or {}
                except Exception:
                    cur = {}
            else:
                cur = {}
            merged = dict(cur)
            for k, v in new_obj.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged

        merged_jg = _merge_yaml(jg_path, jg_out)   
        merged_lp = _merge_yaml(lp_path, lookup_suggest)

        with jg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(merged_jg, fh, sort_keys=True, default_flow_style=False)
        with lp_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(merged_lp, fh, sort_keys=True, default_flow_style=False)

        print(f"\nWrote suggestions into:\n  - {jg_path}\n  - {lp_path}")


if __name__ == "__main__":
    main()
