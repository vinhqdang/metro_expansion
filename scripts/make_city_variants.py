"""Generate single-line city variants from the upstream mo-tndp city folders.

A variant directory data/cities/<name>_<variant>/ contains:
  - config.txt: generated here from the upstream config.txt;
  - od.txt, price_groups_*.txt, average_house_price_gid.txt: symlinks to the
    upstream city folder (external/tabular-tndp/envs/mo-tndp/cities/<name>).

Variants (see design section 4):
  - line1_only: keeps existing line index 0 only, and excluded_od_segments[0:4]
    (Xi'an only; Line 1 corridor exclusions are dropped for the Line 2 corridor
    to be available again). Line 1 is the first line in config.txt.
  - line2_only: keeps existing line index 1 only, and excluded_od_segments[4:8].

Amsterdam has no excluded_od_segments, so its variants keep an empty list.
The upstream external/ tree is only read, never written.

Usage:
    python3.13 scripts/make_city_variants.py xian line1_only
    python3.13 scripts/make_city_variants.py --all
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_CITIES = REPO_ROOT / "external" / "tabular-tndp" / "envs" / "mo-tndp" / "cities"
VARIANT_ROOT = REPO_ROOT / "data" / "cities"

CITIES = ("xian", "amsterdam")
VARIANTS = ("line1_only", "line2_only")

# Index of the single existing line kept in each variant.
LINE_INDEX = {"line1_only": 0, "line2_only": 1}
# Slice of excluded_od_segments kept in each variant (only cities that define them).
EXCLUDED_SLICE = {
    "xian": {"line1_only": slice(0, 4), "line2_only": slice(4, 8)},
}
LINE_KEYS = ("existing_lines", "existing_lines_full")
SHARED_FILES = ("od.txt", "average_house_price_gid.txt")


def _read_upstream_config(upstream_dir: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(upstream_dir / "config.txt")
    return dict(parser["config"])


def make_variant(name: str, variant: str, out_root: Path = VARIANT_ROOT) -> Path:
    """Create data/cities/<name>_<variant>/ and return its path."""
    if name not in CITIES:
        raise ValueError(f"unknown city {name!r}; expected one of {CITIES}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    upstream_dir = UPSTREAM_CITIES / name
    out_dir = out_root / f"{name}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_upstream_config(upstream_dir)
    idx = LINE_INDEX[variant]
    lines = []
    for key, value in raw.items():
        if key in LINE_KEYS:
            kept = [json.loads(value)[idx]]
            lines.append(f"{key} = {json.dumps(kept)}")
        elif key == "excluded_od_segments":
            excluded = json.loads(value)
            sl = EXCLUDED_SLICE.get(name, {}).get(variant)
            kept = excluded[sl] if sl is not None else excluded
            lines.append(f"{key} = {json.dumps(kept)}")
        else:
            lines.append(f"{key} = {value}")
    config_text = "[config]\n" + "\n".join(lines) + "\n"
    (out_dir / "config.txt").write_text(config_text)

    # Symlink the data files that do not change between variants.
    shared = list(SHARED_FILES) + sorted(p.name for p in upstream_dir.glob("price_groups_*.txt"))
    for fname in shared:
        link = out_dir / fname
        target = os.path.abspath(upstream_dir / fname)
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == target:
                continue
            link.unlink()
        os.symlink(target, link)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("city", nargs="?", choices=CITIES)
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    parser.add_argument("--all", action="store_true", help="create every city/variant pair")
    args = parser.parse_args()

    pairs = [(c, v) for c in CITIES for v in VARIANTS] if args.all else [(args.city, args.variant)]
    if not args.all and (args.city is None or args.variant is None):
        parser.error("give a city and a variant, or --all")
    for name, variant in pairs:
        print(make_variant(name, variant))


if __name__ == "__main__":
    main()
