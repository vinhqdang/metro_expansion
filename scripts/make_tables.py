#!/usr/bin/env python3.13
"""Write LaTeX tables and manuscript/generated/numbers.tex from results/summary/*.json.

Outputs (booktabs, no vertical rules):
  tab_main_<city_variant>.tex        methods x [policy EUM, policy HV, archive HV, IGD+ (policy), wall-clock]
  tab_objectives_<city_variant>.tex  per-objective values at the four weight settings
  tab_tests_<city_variant>.tex       primary comparisons against RL (Holm within each metric family)
  tab_transfer_<src>_to_<tgt>.tex    transfer conditions against scratch, per budget
  numbers.tex                        one \\newcommand per quoted number (names use letters only)

Rounding: indicators 3 decimals, raw objectives 4 decimals, wall-clock 1 decimal, p-values
3 decimals or $<$0.001. Undefined values are written as -- in tables and n/a in numbers.tex.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

SUMMARY_SCHEMA = "mnep.summary.v1"
TRANSFER_SCHEMA = "mnep.transfer.v1"
REF = "rl_full"

DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
# Short tokens for macro names (letters only).
SHORT = {
    "rl_full": "RL",
    "rl_single_objective": "RLDemand",
    "rl_flat_encoder": "RLFlat",
    "rl_noprior": "RLNoPrior",
    "nsga2": "NSGA",
    "ga_scalar_matched": "GAMatched",
    "ga_scalar": "GAFull",
    "ga_demand": "GADemand",
    "ga_ggi2": "GAGGI",
    "tabular_eff": "TabDemand",
    "tabular_ggi2": "TabGGI",
    "greedy": "Greedy",
    "random": "Random",
}
CITY_NAMES = {"xian": "Xi'an", "amsterdam": "Amsterdam", "hcmc": "Ho Chi Minh City"}
VARIANT_TEXT = {"full": "full network", "line1_only": "Line 1 only"}
COND_LABELS = {
    "scratch": "Scratch",
    "full": "Full transfer",
    "encoder_only": "Encoder only",
    "heads_only": "Heads only",
    "full_frozen_enc": "Full, frozen encoder",
    "random_frozen_enc": "Random frozen encoder",
}
SETTING_TEXT = {
    "vertex_demand": r"Vertex demand, $w=(1,0,0)$",
    "vertex_fairness": r"Vertex fairness, $w=(0,1,0)$",
    "vertex_access": r"Vertex access, $w=(0,0,1)$",
    "mean_all": r"Mean over the 15 weight vectors",
}
SETTING_TOK = {
    "vertex_demand": "VertexDemand",
    "vertex_fairness": "VertexFairness",
    "vertex_access": "VertexAccess",
    "mean_all": "MeanAll",
}
OBJ_TEX = {
    "D_tot": r"$D_{\mathrm{tot}}$",
    "ggi2": r"$\mathrm{GGI}_2$",
    "s_min": r"$s_{\min}$",
    "A_new": r"$A_{\mathrm{new}}$",
}
OBJ_TOK = {"D_tot": "Dtot", "ggi2": "Ggi", "s_min": "Smin", "A_new": "Anew"}
# (metric key in summary, macro token, decimals)
MAIN_METRICS = [
    ("policy_eum", "Eum", 3),
    ("policy_hv", "Hv", 3),
    ("archive_hv", "ArchHv", 3),
    ("archive_eum", "ArchEum", 3),
    ("policy_igd_plus", "Igd", 3),
    ("archive_igd_plus", "ArchIgd", 3),
    ("wall_clock_s", "Wall", 1),
]
TABLE_METRICS = ["policy_eum", "policy_hv", "archive_hv", "policy_igd_plus", "wall_clock_s"]
FAMILY_TOK = {"archive_hv": "ArchHv", "policy_eum": "PolicyEum"}
FAMILY_TEX = {"archive_hv": "Archive HV", "policy_eum": "Policy EUM"}

MISSING_TABLE = "--"
MISSING_MACRO = "n/a"


# ---------------------------------------------------------------- text helpers
_DIGIT_RE = re.compile(r"\d")


def tok(s: str) -> str:
    """CamelCase token of a directory-style name, digits spelled out: 'xian_line1_only' -> 'XianLineOneOnly'."""
    out = []
    for part in re.split(r"[^A-Za-z0-9]+", s):
        if not part:
            continue
        spelled = _DIGIT_RE.sub(lambda m: DIGIT_WORDS[int(m.group(0))], part)
        out.append(spelled[:1].upper() + spelled[1:])
    return "".join(out)


_ESC = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    return "".join(_ESC.get(ch, ch) for ch in str(s))


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def num(x: Any, d: int, missing: str = MISSING_TABLE) -> str:
    """Fixed-decimal number with a math-mode minus sign; `missing` for undefined values."""
    if not _finite(x):
        return missing
    v = round(float(x), d)
    if v == 0:
        v = 0.0
    return f"{v:.{d}f}".replace("-", "$-$")


def pval(p: Any, missing: str = MISSING_TABLE) -> str:
    if not _finite(p):
        return missing
    if float(p) < 0.001:
        return "$<$0.001"
    return f"{float(p):.3f}"


def pm(stat: dict[str, Any] | None, d: int, missing: str = MISSING_TABLE) -> str:
    """mean $\\pm$ sd, or mean alone when sd is undefined (n=1)."""
    if not stat or not _finite(stat.get("mean")):
        return missing
    if not _finite(stat.get("sd")):
        return num(stat["mean"], d, missing)
    return f"{num(stat['mean'], d, missing)} $\\pm$ {num(stat['sd'], d, missing)}"


def ci(stat_ci: list[Any] | None, d: int, missing: str = MISSING_TABLE) -> str:
    if not stat_ci or not _finite(stat_ci[0]) or not _finite(stat_ci[1]):
        return missing
    return f"[{num(stat_ci[0], d, missing)}, {num(stat_ci[1], d, missing)}]"


def city_text(cv: str) -> str:
    city, _, variant = cv.partition("_")
    return f"{CITY_NAMES.get(city, city)}, {VARIANT_TEXT.get(variant, variant)}"


def city_tok(cv: str) -> str:
    """Macro prefix for a city variant; the default 'full' variant is dropped: 'xian_full' -> 'Xian'."""
    base = cv[: -len("_full")] if cv.endswith("_full") else cv
    return tok(base)


def method_tok(m: str) -> str:
    return SHORT.get(m, tok(m))


# ---------------------------------------------------------------- macros
class Macros:
    """Ordered, validated list of \\newcommand definitions."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"invalid LaTeX macro name {name!r}")
        if name in self.names:
            raise ValueError(f"duplicate macro name {name!r}")
        self.names.add(name)
        self.items.append((name, value))

    def add_stat(self, base: str, stat: dict[str, Any] | None, d: int) -> None:
        """base = mean, baseSd, baseCiLo, baseCiHi."""
        mean = stat.get("mean") if stat else None
        sd = stat.get("sd") if stat else None
        cl = stat.get("ci95", [None, None]) if stat else [None, None]
        self.add(base, num(mean, d, MISSING_MACRO))
        self.add(base + "Sd", num(sd, d, MISSING_MACRO))
        self.add(base + "CiLo", num(cl[0], d, MISSING_MACRO))
        self.add(base + "CiHi", num(cl[1], d, MISSING_MACRO))

    def write(self, path: Path, header: str) -> None:
        lines = [f"% {header}", "% Generated by scripts/make_tables.py. Do not edit by hand."]
        for name, value in self.items:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- tables
def _table_wrap(caption: str, label: str, body: str, font: str = "") -> str:
    return (
        "% Generated by scripts/make_tables.py. Do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{font}"
        f"{body}"
        "\\end{table}\n"
    )


def table_main(cv: str, summ: dict[str, Any]) -> tuple[str, str]:
    cols = ["Method", "$n$", "Policy EUM", "Policy HV", "Archive HV", "IGD$^+$ (policy)", "Wall-clock (s)"]
    rows = []
    for m, s in summ["methods"].items():
        label = latex_escape(s["label"]) + ("$^\\dagger$" if s["single_point"] else "")
        cells = [label, str(s["n_seeds"])]
        for key in TABLE_METRICS:
            d = 1 if key == "wall_clock_s" else 3
            cells.append(pm(s["metrics"][key], d))
        rows.append(" & ".join(cells) + " \\\\")
    body = (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        + " & ".join(cols) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    caption = (
        f"Main results, {latex_escape(city_text(cv))}. Mean $\\pm$ standard deviation over the "
        "seeds in column $n$. Policy metrics use the deployable greedy front; archive metrics use "
        "every evaluated line. $\\dagger$: single-line method."
    )
    return _table_wrap(caption, f"tab:main_{cv}", body, font="\\small\n"), body


def table_objectives(cv: str, summ: dict[str, Any]) -> str:
    cols = ["Method"] + [OBJ_TEX[k] for k in ("D_tot", "ggi2", "s_min", "A_new")]
    blocks = []
    any_available = False
    for ws in summ["weight_settings"]:
        name = ws["name"]
        blocks.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{SETTING_TEXT[name]}}}}} \\\\\n\\midrule")
        for m, s in summ["methods"].items():
            obj = s["objectives"][name]
            label = latex_escape(s["label"]) + ("$^\\dagger$" if s["single_point"] else "")
            if obj["available"]:
                any_available = True
            cells = [label] + [pm(obj[k], 4) for k in ("D_tot", "ggi2", "s_min", "A_new")]
            blocks.append(" & ".join(cells) + " \\\\")
        blocks.append("\\addlinespace")
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        + " & ".join(cols) + " \\\\\n\\midrule\n"
        + "\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    caption = (
        f"Raw objective values at the three vertex weights and averaged over the 15 weight vectors, {latex_escape(city_text(cv))}. "
        "Mean $\\pm$ standard deviation over seeds; each value is taken from the deployable front point "
        "with the matching weight vector. $\\dagger$: single-line method, the same line at every setting."
    )
    if not any_available:
        caption += " The last block averages the 15 deployable points."
    return _table_wrap(caption, f"tab:objectives_{cv}", body, font="\\footnotesize\n")


def table_tests(cv: str, summ: dict[str, Any]) -> str:
    tests = summ["tests_vs"]
    rows = []
    for fam, block in tests["families"].items():
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{FAMILY_TEX[fam]}}}}} \\\\\n\\midrule")
        for r in block["rows"]:
            comp = f"{latex_escape(summ['methods'][REF]['label'])} vs {latex_escape(r['label'])}"
            if r["tested"]:
                rows.append(" & ".join([
                    comp,
                    str(r["n_seeds"]),
                    num(r["mean_diff"], 3),
                    ci(r["diff_ci95"], 3),
                    num(r["a12"], 3),
                    pval(r["p"]),
                    pval(r["p_holm"]),
                ]) + " \\\\")
            else:
                rows.append(f"{comp} & {r['n_seeds']} & \\multicolumn{{5}}{{l}}{{not tested ($n<3$)}} \\\\")
        rows.append("\\addlinespace")
    body = (
        "\\begin{tabular}{lrrlrrr}\n\\toprule\n"
        "Comparison & $n$ & Mean difference & 95\\% CI & $\\hat{A}_{12}$ & $p$ & $p_{\\text{Holm}}$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    caption = (
        f"Primary comparisons against RL (ours), {latex_escape(city_text(cv))}. Two-sided Mann-Whitney "
        "test with Vargha-Delaney $\\hat{A}_{12}$ and a 95\\% bootstrap interval of the mean difference "
        "(RL minus comparator). Holm correction within each metric family."
    )
    return _table_wrap(caption, f"tab:tests_{cv}", body, font="\\footnotesize\n")


def table_transfer(pair: str, summ: dict[str, Any]) -> str:
    src, _, tgt = pair.partition("_to_")
    pair_text = f"{CITY_NAMES.get(src, src)} to {CITY_NAMES.get(tgt, tgt)}"
    blocks = []
    for budget, blk in summ["budgets"].items():
        blocks.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{latex_escape(budget.capitalize())} budget}}}} \\\\\n\\midrule")
        tests = {(t["condition"], t["outcome"]): t for t in blk["tests"]}
        for cond in ("scratch", "full", "encoder_only", "heads_only", "full_frozen_enc", "random_frozen_enc"):
            if cond not in blk["conditions"]:
                continue
            c = blk["conditions"][cond]
            cells = [COND_LABELS[cond], str(c["n"]),
                     pm(c["final_eum"], 3), pm(c["auc_eum"], 3)]
            for outcome in ("final_eum", "auc_eum"):
                t = tests.get((cond, outcome))
                if t is None:
                    cells += ["--", "--"]
                else:
                    cells += [f"{num(t['hl_shift'], 3)} {ci(t['ci95'], 3)}", pval(t["p_holm"])]
            blocks.append(" & ".join(cells) + " \\\\")
        blocks.append("\\addlinespace")
    body = (
        "\\begin{tabular}{lrcccccc}\n\\toprule\n"
        "Condition & $n$ & Final EUM & Mean EUM over training & "
        "$\\Delta$ final [95\\% CI] & $p_{\\text{Holm}}$ & $\\Delta$ mean EUM [95\\% CI] & $p_{\\text{Holm}}$ \\\\\n"
        "\\midrule\n" + "\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    caption = (
        f"Transfer, {latex_escape(pair_text)}. Paired by seed; $\\Delta$ is the Hodges-Lehmann shift "
        "from scratch. Holm correction over all contrasts of one budget."
    )
    return _table_wrap(caption, f"tab:transfer_{pair}", body, font="\\footnotesize\\setlength{\\tabcolsep}{3pt}\n")


# ---------------------------------------------------------------- numbers
def main_macros(cv: str, summ: dict[str, Any], mac: Macros) -> None:
    cp = city_tok(cv)
    for m, s in summ["methods"].items():
        mt = cp + method_tok(m)
        mac.add(mt + "N", str(s["n_seeds"]))
        for key, tk, d in MAIN_METRICS:
            mac.add_stat(mt + tk, s["metrics"][key], d)
        mac.add(mt + "Npoints", str(s["n_points"]))
        mac.add(mt + "NA", str(s["n_na"]))
        for ws in summ["weight_settings"]:
            name = ws["name"]
            for k, ot in OBJ_TOK.items():
                mac.add_stat(mt + SETTING_TOK[name] + ot, s["objectives"][name][k], 4)
    tests = summ["tests_vs"]
    if REF in summ["methods"]:
        rt = cp + method_tok(REF)
        for fam, block in tests["families"].items():
            for r in block["rows"]:
                if not r["tested"]:
                    continue
                base = rt + "vs" + method_tok(r["method"]) + FAMILY_TOK[fam]
                mac.add(base + "Diff", num(r["mean_diff"], 3, MISSING_MACRO))
                mac.add(base + "DiffLo", num(r["diff_ci95"][0], 3, MISSING_MACRO))
                mac.add(base + "DiffHi", num(r["diff_ci95"][1], 3, MISSING_MACRO))
                mac.add(base + "Vda", num(r["a12"], 3, MISSING_MACRO))
                mac.add(base + "P", pval_macro(r["p"]))
                mac.add(base + "PHolm", pval_macro(r["p_holm"]))


def pval_macro(p: Any) -> str:
    if not _finite(p):
        return MISSING_MACRO
    return "$<$0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def transfer_macros(pair: str, summ: dict[str, Any], mac: Macros) -> None:
    pt = tok(pair)
    for budget, blk in summ["budgets"].items():
        tests = {(t["condition"], t["outcome"]): t for t in blk["tests"]}
        for cond, c in blk["conditions"].items():
            base = pt + tok(cond) + "Budget" + tok(budget)
            mac.add(base + "N", str(c["n"]))
            mac.add(base + "FinalEum", num(c["final_eum"]["mean"], 3, MISSING_MACRO))
            mac.add(base + "FinalEumSd", num(c["final_eum"]["sd"], 3, MISSING_MACRO))
            mac.add(base + "AucEum", num(c["auc_eum"]["mean"], 3, MISSING_MACRO))
            mac.add(base + "AucEumSd", num(c["auc_eum"]["sd"], 3, MISSING_MACRO))
            for outcome, tk in (("final_eum", "Final"), ("auc_eum", "Auc")):
                t = tests.get((cond, outcome))
                if t is None:
                    continue
                b2 = base + tk
                mac.add(b2 + "Delta", num(t["hl_shift"], 3, MISSING_MACRO))
                mac.add(b2 + "DeltaLo", num(t["ci95"][0], 3, MISSING_MACRO))
                mac.add(b2 + "DeltaHi", num(t["ci95"][1], 3, MISSING_MACRO))
                mac.add(b2 + "P", pval_macro(t["p"]))
                mac.add(b2 + "PHolm", pval_macro(t["p_holm"]))


# ---------------------------------------------------------------- driver
def load_summaries(summary_dir: Path) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    main, transfer = [], []
    for f in sorted(Path(summary_dir).glob("*.json")):
        obj = json.loads(f.read_text())
        if obj.get("schema") == SUMMARY_SCHEMA:
            main.append((obj["city_variant"], obj))
        elif obj.get("schema") == TRANSFER_SCHEMA:
            transfer.append((obj["pair"], obj))
        else:
            print(f"skipping {f}: unknown schema {obj.get('schema')!r}", file=sys.stderr)
    return main, transfer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, default=REPO / "results" / "summary", help="summary directory")
    ap.add_argument("--out", type=Path, default=REPO / "manuscript" / "generated", help="output directory")
    args = ap.parse_args(argv)

    main_list, transfer_list = load_summaries(args.summary)
    if not main_list and not transfer_list:
        raise FileNotFoundError(f"no summary JSON files in {args.summary}")
    args.out.mkdir(parents=True, exist_ok=True)
    mac = Macros()
    written = []
    for cv, summ in main_list:
        for name, text in (
            (f"tab_main_{cv}.tex", table_main(cv, summ)[0]),
            (f"tab_objectives_{cv}.tex", table_objectives(cv, summ)),
            (f"tab_tests_{cv}.tex", table_tests(cv, summ)),
        ):
            (args.out / name).write_text(text)
            written.append(name)
        main_macros(cv, summ, mac)
    for pair, summ in transfer_list:
        name = f"tab_transfer_{pair}.tex"
        (args.out / name).write_text(table_transfer(pair, summ))
        written.append(name)
        transfer_macros(pair, summ, mac)
    mac.write(args.out / "numbers.tex", "Quoted numbers (one macro per value).")
    written.append("numbers.tex")
    for name in written:
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
