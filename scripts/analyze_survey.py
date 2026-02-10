#!/usr/bin/env python3
"""Reproducible cross-sectional analysis of CPA pathway survey.

Usage:
  python scripts/analyze_survey.py --data-file "Alternative ...csv" --output-dir artifact_outputs
"""
import argparse
import csv
import json
import os
import re
from collections import Counter
from statistics import mean

DV_TERMS = ["intent", "plan", "graduate", "macc", "master", "enroll", "desire", "likely"]
PERCEPTION_TERMS = ["150", "credit", "hours", "requirement", "cpa", "pathway", "barrier", "cost", "time", "availability", "aware"]


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def load_qualtrics(path: str):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    qids = rows[0]
    qtexts = rows[1]
    import_meta = rows[2]
    data = rows[3:]
    mapping = {
        qid: {
            "question_text": clean_text(qtexts[i]) if i < len(qtexts) else "",
            "import_meta": import_meta[i] if i < len(import_meta) else "",
            "index": i,
        }
        for i, qid in enumerate(qids)
    }
    return qids, qtexts, data, mapping


def nonempty_values(data, idx):
    return [row[idx].strip() for row in data if idx < len(row) and row[idx].strip()]


def overlap_nonempty(data, idx_a, idx_b):
    n = 0
    for row in data:
        va = row[idx_a].strip() if idx_a < len(row) else ""
        vb = row[idx_b].strip() if idx_b < len(row) else ""
        if va and vb:
            n += 1
    return n


def response_profile(values):
    c = Counter(values)
    return {"n": len(values), "unique": len(c), "counts": c}


def is_open_ended(question_text: str, profile: dict):
    qt = question_text.lower()
    if any(x in qt for x in ["please explain", "please describe", "briefly describe", "share any thoughts"]):
        return True
    return profile["unique"] > 20


def maybe_ordered(profile: dict):
    return 2 <= profile["unique"] <= 7


def dv_score(question_text: str):
    t = question_text.lower()
    score = sum(2 for k in DV_TERMS if k in t)
    if "graduate" in t or "macc" in t or "master" in t:
        score += 4
    if "desire to pursue a graduate" in t:
        score += 6
    if "if you had known" in t:
        score -= 1
    if "undergraduate student or graduate student" in t:
        score -= 5
    return score


def likert_numeric(value: str):
    v = clean_text(value).lower()
    exact = {
        "strongly disagree": 1,
        "somewhat disagree": 2,
        "neither agree nor disagree": 3,
        "somewhat agree": 4,
        "strongly agree": 5,
        "extremely unlikely": 1,
        "somewhat unlikely": 2,
        "neither likely nor unlikely": 3,
        "somewhat likely": 4,
        "extremely likely": 5,
        "significantly decreased desire": 1,
        "decreased desire": 2,
        "no change in desire": 3,
        "increased desire": 4,
        "significantly increased desire": 5,
        "very negative": 1,
        "somewhat negative": 2,
        "neutral": 3,
        "somewhat positive": 4,
        "very positive": 5,
        "no": 0,
        "yes": 1,
    }
    if v in exact:
        return exact[v]
    if v.isdigit():
        return float(v)
    return None


def transpose(m):
    return [list(row) for row in zip(*m)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def inverse(matrix):
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-10:
            raise ValueError("Singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ols(y, xcols):
    X = [[1.0] + [col[i] for col in xcols] for i in range(len(y))]
    Y = [[v] for v in y]
    Xt = transpose(X)
    beta = [b[0] for b in matmul(matmul(inverse(matmul(Xt, X)), Xt), Y)]
    preds = [sum(b * xv for b, xv in zip(beta, row)) for row in X]
    ybar = sum(y) / len(y)
    sst = sum((yi - ybar) ** 2 for yi in y)
    ssr = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2 = 1 - ssr / sst if sst > 0 else 0
    return beta, r2


def write_dv_chart_svg(dv_counter, out_path):
    items = dv_counter.most_common()
    if not items:
        return
    width = 920
    height = 420
    margin_left = 60
    margin_bottom = 90
    chart_w = width - margin_left - 30
    chart_h = height - 40 - margin_bottom
    max_count = max(v for _, v in items)
    bar_w = chart_w / max(len(items), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:16px;font-weight:bold}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" class="title">DV distribution: Graduate intent/desire (Q52)</text>',
    ]
    # axes
    y0 = height - margin_bottom
    x0 = margin_left
    lines.append(f'<line x1="{x0}" y1="40" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-20}" y2="{y0}" stroke="black"/>')

    for i, (label, count) in enumerate(items):
        bh = (count / max_count) * (chart_h - 10)
        x = x0 + i * bar_w + 10
        y = y0 - bh
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-20:.1f}" height="{bh:.1f}" fill="#2a6fdb"/>')
        lines.append(f'<text x="{x + (bar_w-20)/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{count}</text>')
        short = label[:22] + "…" if len(label) > 23 else label
        lines.append(f'<text x="{x + (bar_w-20)/2:.1f}" y="{y0+16}" text-anchor="middle">{short}</text>')

    lines.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="Alternative CPA Pathways Survey_December 31, 2025_09.45.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    report_file = os.path.join(args.output_dir, "report.md")
    details_file = os.path.join(args.output_dir, "analysis_details.json")
    fig_file = os.path.join(args.output_dir, "dv_distribution.svg")

    qids, qtexts, data, mapping = load_qualtrics(args.data_file)

    dv_candidates = []
    for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
        profile = response_profile(nonempty_values(data, i))
        if profile["n"] == 0 or is_open_ended(qtext, profile) or not maybe_ordered(profile):
            continue
        score = dv_score(qtext)
        if score > 0:
            dv_candidates.append((score, profile["n"], i, qid, clean_text(qtext)))
    dv_candidates.sort(reverse=True)
    _, _, dv_idx, dv_qid, dv_text = dv_candidates[0]

    perception_candidates = []
    for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
        if i == dv_idx:
            continue
        qt = clean_text(qtext)
        ql = qt.lower()
        kscore = sum(1 for k in PERCEPTION_TERMS if k in ql)
        profile = response_profile(nonempty_values(data, i))
        if kscore < 2 or profile["n"] == 0 or is_open_ended(qtext, profile) or not maybe_ordered(profile):
            continue
        if "rank" in ql or "how likely are you to pursue a cpa license" in ql:
            continue
        overlap = overlap_nonempty(data, dv_idx, i)
        if overlap >= 30:
            perception_candidates.append((kscore, overlap, profile["n"], i, qid, qt))

    perception_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = []
    for item in perception_candidates:
        ql = item[5].lower()
        if any(t in ql for t in ["overall perception", "impacted your desire", "aware", "alternative pathway", "credit hour", "requirements"]):
            selected.append(item)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected = perception_candidates[:3]

    y = []
    X_cols = [[], []]
    dv_dist = Counter()
    per1_idx, per2_idx, aware_idx = selected[0][3], selected[1][3], selected[2][3]

    for row in data:
        vals = [(row[idx].strip() if idx < len(row) else "") for idx in [dv_idx, per1_idx, per2_idx, aware_idx]]
        if any(v == "" for v in vals):
            continue
        dv_num, p1_num, p2_num, aware_num = [likert_numeric(v) for v in vals]
        if None in (dv_num, p1_num, p2_num, aware_num):
            continue
        y.append(float(dv_num))
        X_cols[0].append((p1_num + p2_num) / 2.0)
        X_cols[1].append(float(aware_num))
        dv_dist[vals[0]] += 1

    beta, r2 = ols(y, X_cols)

    os.makedirs(args.output_dir, exist_ok=True)
    write_dv_chart_svg(dv_dist, fig_file)

    details = {
        "dv": {"qid": dv_qid, "question_text": dv_text},
        "qid_mapping": mapping,
        "perception_predictors": [
            {"qid": s[4], "question_text": s[5], "keyword_score": s[0], "overlap_with_dv": s[1], "non_missing": s[2]}
            for s in selected
        ],
        "coefficients": {"intercept": beta[0], "perception_index": beta[1], "awareness_yes": beta[2], "r_squared": r2},
        "descriptives": {
            "n_total_responses": len(data),
            "n_model": len(y),
            "dv_distribution": dict(dv_dist),
            "dv_mean": mean(y),
            "perception_index_mean": mean(X_cols[0]),
            "aware_share": mean(X_cols[1]),
        },
        "visualization": {"file": os.path.basename(fig_file), "description": "Bar chart of DV category frequencies."},
    }

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    lines = [
        "# CPA Pathway Survey: Cross-Sectional Association Analysis",
        "",
        "## Question",
        "How are students’ perceptions of the CPA 150-credit-hour requirement (and alternative-pathway framing) associated with their stated intent to pursue graduate accounting education?",
        "",
        "## Data handling and Qualtrics headers",
        "- Used a parser that treats Qualtrics row 1 as QIDs, row 2 as human-readable question text, and row 3 as import metadata; respondent data starts on row 4.",
        "- Preserved a QID→question-text mapping in `analysis_details.json` for interpretability.",
        "",
        "## Programmatic variable selection",
        f"- **Primary DV selected via header scan:** `{dv_qid}` — {dv_text}",
        "- **Perception predictors** (header-scan based):",
    ]
    lines.extend([f"  - `{s[4]}` — {s[5]}" for s in selected])
    lines.extend([
        "",
        "## Simple descriptives",
        f"- Model sample (complete cases): **{len(y)}** respondents (out of {len(data)} rows).",
        f"- DV mean (1=significantly decreased desire ... 5=significantly increased desire): **{mean(y):.2f}**.",
        f"- Perception index mean: **{mean(X_cols[0]):.2f}**.",
        f"- Awareness (`Yes`) share: **{mean(X_cols[1]):.2%}**.",
        "",
        "## Association model",
        "Because the selected DV is an ordered 5-level intent/desire item, a simple **OLS association model** was fit as a lightweight approximation:",
        "",
        "`DV_numeric = b0 + b1*(perception_index) + b2*(aware_before_survey)`",
        "",
        f"- Intercept (b0): **{beta[0]:.3f}**",
        f"- Perception index (b1): **{beta[1]:.3f}**",
        f"- Awareness yes=1 (b2): **{beta[2]:.3f}**",
        f"- R²: **{r2:.3f}**",
        "",
        "## Visualization",
        "- `dv_distribution.svg`: bar chart of the selected DV categories (see artifact output folder).",
        "",
        "## Limitations",
        "- This is **cross-sectional** survey data; each respondent is observed once, so results are associational and not causal.",
        "- The outcome is **self-reported intent/desire**, not observed later enrollment behavior.",
        "- Use interpretation as “associated with / related to,” not causal language.",
    ])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {report_file}")
    print(f"Wrote: {details_file}")
    print(f"Wrote: {fig_file}")


if __name__ == "__main__":
    main()
