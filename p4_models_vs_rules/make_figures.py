"""Build every figure for the models-vs-rules paper from the source notebooks in notebooks/.

No data are reconstructed or simulated. Each value plotted is parsed from the
printed output of a notebook cell (the notebooks' own copy-paste summary blocks),
so every point traces to a specific notebook and cell:

  Figure 1  skill by test year     notebooks/moonshot_ml_sandbox_phase2_logistic.ipynb
                                   summary cell, block "[skill by test year, M1 / M2]"
  Figure 2  Sharpe-difference CIs  notebooks/moonshot_ml_sandbox_phase3a_portfolios_v2.ipynb
                                   ("[performance]", "[vs baseline]", "4. stricter CIs")
                                   notebooks/moonshot_ml_sandbox_phase3b_model_vs_twin.ipynb
                                   ("[model minus twin: Sharpe diff]")

Usage:  python3 make_figures.py      (writes figures/*.pdf and figures/*_values.csv)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "notebooks"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def notebook_text(name):
    """All printed text outputs of a notebook, concatenated in cell order."""
    nb = json.loads((SRC / name).read_text())
    out = []
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            if "text" in o:
                out.append("".join(o["text"]))
    return "\n".join(out)


def block_after(text, header, n_lines):
    """The n_lines non-empty lines that follow the LAST occurrence of `header`."""
    start = text.rindex(header) + len(header)
    lines = [l for l in text[start:].splitlines() if l.strip()]
    return lines[:n_lines]


# ---------------------------------------------------------------------------
# Figure 1: out-of-sample skill by test year (Phase 2, M1)
# ---------------------------------------------------------------------------
p2 = notebook_text("moonshot_ml_sandbox_phase2_logistic.ipynb")
rows = []
for line in block_after(p2, "[skill by test year, M1 / M2]", 4):
    event = re.match(r"\s*(Outperform|Top performer|Crash|High vol)", line).group(1)
    for yr, m1, m2 in re.findall(r"(\d{4}):([+-]\d\.\d+)/([+-]\d\.\d+)", line):
        rows.append(dict(event=event, test_year=int(yr), skill_M1=float(m1), skill_M2=float(m2)))
skill = pd.DataFrame(rows)
assert len(skill) == 48, f"expected 4 events x 12 test years, parsed {len(skill)}"
skill.to_csv(FIG / "fig1_skill_by_year_values.csv", index=False)

STYLE = {"Outperform": dict(color="0.0", ls="-", marker="o"),
         "Top performer": dict(color="0.45", ls="--", marker="s"),
         "Crash": dict(color="0.2", ls=":", marker="^"),
         "High vol": dict(color="0.0", ls="-", marker="D")}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, events in [(a1, ["Outperform", "Top performer", "Crash"]), (a2, ["High vol"])]:
    ax.axvspan(2021.5, 2025.5, color="0.9", lw=0, zorder=0)
    for e in events:
        s = skill[skill.event == e].sort_values("test_year")
        ax.plot(s.test_year, s.skill_M1, ms=4, lw=1.2, label=e, **STYLE[e])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("test year")
    ax.set_xticks(range(2014, 2026, 2))
a1.set_ylabel("skill vs intercept-only model (M1)")
a1.legend(frameon=False, fontsize=8, loc="upper left")
a2.legend(frameon=False, fontsize=8, loc="lower left")
a2.set_ylim(bottom=0)
a1.text(2023.5, a1.get_ylim()[1] * 0.92, "2022+", ha="center", fontsize=8, color="0.3")
a2.text(2023.5, a2.get_ylim()[1] * 0.92, "2022+", ha="center", fontsize=8, color="0.3")
fig.tight_layout()
fig.savefig(FIG / "fig1_skill_by_year.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Sharpe differences with bootstrap CIs (Phase 3a vs baseline; 3b vs twin)
# ---------------------------------------------------------------------------
p3a = notebook_text("moonshot_ml_sandbox_phase3a_portfolios_v2.ipynb")
perf = {}
for line in block_after(p3a, "[performance]  (invested months only)", 7)[2:]:
    name = re.match(r"(.+?)\s+[+-]\d", line).group(1).strip()
    nums = [float(x) for x in re.findall(r"[+-]\d+\.\d+", line)]
    perf[name] = dict(cagr=nums[0], vol=nums[1], sharpe=nums[2])
vsb = {}
for line in block_after(p3a, "[vs baseline]", 6)[2:]:
    name = re.match(r"(.+?)\s+[+-]\d", line).group(1).strip()
    nums = [float(x) for x in re.findall(r"[+-]\d+\.\d+", line)]
    vsb[name] = dict(lo95=nums[7], hi95=nums[8])
strict = {}
for line in block_after(p3a, "4. stricter CIs for 4 portfolios tested at once (98.75% instead of 95%):", 4):
    m = re.match(r"\s*(.+?)\s+active .*Sharpe diff \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", line)
    strict[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))

p3b = notebook_text("moonshot_ml_sandbox_phase3b_model_vs_twin.ipynb")
twin = []
for line in block_after(p3b, "[model minus twin: Sharpe diff]", 4):
    m = re.match(r"\s*(.+?)\s+vs (.+?)\s+([+-]\d\.\d+)\s+95% \[([+-]\d\.\d+), ([+-]\d\.\d+)\]"
                 r"\s+98\.33% \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", line)
    twin.append(dict(label=f"{m.group(1).strip()} vs {m.group(2).strip()}",
                     pt=float(m.group(3)), lo95=float(m.group(4)), hi95=float(m.group(5)),
                     loB=float(m.group(6)), hiB=float(m.group(7)), level_B="98.33%",
                     panel="model minus twin"))

base = perf["Baseline"]["sharpe"]
vs_base = [dict(label=f"{p} vs baseline", pt=round(perf[p]["sharpe"] - base, 3),
                lo95=vsb[p]["lo95"], hi95=vsb[p]["hi95"], loB=strict[p][0], hiB=strict[p][1],
                level_B="98.75%", panel="model minus baseline")
           for p in ["Pick outperformers", "Pick big winners", "Avoid crashes", "Avoid wild movers"]]
ci = pd.DataFrame(vs_base + twin)
ci.to_csv(FIG / "fig2_sharpe_diff_values.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 3.3))
n = len(ci)
for i, r in ci.reset_index(drop=True).iterrows():
    y = n - 1 - i
    ax.plot([r.loB, r.hiB], [y, y], color="0.45", lw=1.0)
    ax.plot([r.lo95, r.hi95], [y, y], color="0.1", lw=4.0, solid_capstyle="butt")
    ax.plot(r.pt, y, "o", ms=6, mfc="white", mec="0.0", mew=1.3, zorder=3)
ax.axvline(0, color="0.2", lw=0.8)
ax.axhline(len(twin) - 0.5, color="0.6", lw=0.6, ls="--")
labels = [l.replace("Twin: ", "comparator: ").replace("Ref: ", "ref: ").replace(" - ", " $-$ ")
          for l in ci["label"]]
labels = [l + "  (reference)" if "ref:" in l else l for l in labels]
ax.set_yticks(range(n))
ax.set_yticklabels(labels[::-1], fontsize=8)
ax.set_xlabel("difference in annualised Sharpe ratio (gross of costs)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "fig2_sharpe_diff.pdf")
plt.close(fig)

print("wrote:", *sorted(p.name for p in FIG.iterdir()), sep="\n  ")
print(ci.to_string(index=False))
