#!/usr/bin/env python3
"""Reproducible cross-sectional analysis of CPA pathway survey."""
import csv
import json
import math
import os
import re
from collections import Counter
from statistics import mean

DATA_FILE = "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
REPORT_FILE = "outputs/report.md"
DETAILS_FILE = "outputs/analysis_details.json"

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
    out = []
    for row in data:
        if idx < len(row):
            v = row[idx].strip()
            if v:
                out.append(v)
    return out


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
    # Keep compact categorical variables, exclude ranking fields with many integer-coded options.
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


def normalize_for_scoring(val: str):
    return clean_text(val).lower()


def likert_numeric(value: str):
    v = normalize_for_scoring(value)

    exact = {
        "strongly disagree": 1,
        "disagree": 2,
        "somewhat disagree": 2,
        "neither agree nor disagree": 3,
        "neutral": 3,
        "neither likely nor unlikely": 3,
        "neither attractive nor unattractive": 3,
        "neither satisfied nor dissatisfied": 3,
        "agree": 4,
        "somewhat agree": 4,
        "strongly agree": 5,
        "extremely unlikely": 1,
        "somewhat unlikely": 2,
        "somewhat likely": 4,
        "extremely likely": 5,
        "not at all attractive": 1,
        "somewhat unattractive": 2,
        "somewhat attractive": 4,
        "very attractive": 5,
        "extremely dissatisfied": 1,
        "somewhat dissatisfied": 2,
        "somewhat satisfied": 4,
        "extremely satisfied": 5,
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

    # Pattern-based fallback.
    if "significant" in v and "increase" in v:
        return 5
    if "increase" in v:
        return 4
    if "no change" in v:
        return 3
    if "decrease" in v:
        return 2
    if "very positive" in v:
        return 5
    if "somewhat positive" in v:
        return 4
    if "neutral" in v:
        return 3
    if "somewhat negative" in v:
        return 2
    if "very negative" in v:
        return 1

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
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ols(y, xcols):
    X = [[1.0] + [col[i] for col in xcols] for i in range(len(y))]
    Y = [[v] for v in y]
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    XtX_inv = inverse(XtX)
    XtY = matmul(Xt, Y)
    beta = matmul(XtX_inv, XtY)
    beta = [b[0] for b in beta]

    preds = []
    for row in X:
        preds.append(sum(b * xv for b, xv in zip(beta, row)))

    ybar = sum(y) / len(y)
    sst = sum((yi - ybar) ** 2 for yi in y)
    ssr = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
    r2 = 1 - ssr / sst if sst > 0 else 0
    return beta, r2


def main():
    qids, qtexts, data, mapping = load_qualtrics(DATA_FILE)

    # Choose DV from question text scan.
    dv_candidates = []
    for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
        profile = response_profile(nonempty_values(data, i))
        if profile["n"] == 0 or is_open_ended(qtext, profile) or not maybe_ordered(profile):
            continue
        score = dv_score(qtext)
        if score <= 0:
            continue
        dv_candidates.append((score, profile["n"], i, qid, clean_text(qtext)))
    dv_candidates.sort(reverse=True)
    dv = dv_candidates[0]
    _, _, dv_idx, dv_qid, dv_text = dv

    # Perception candidates.
    perception_candidates = []
    for i, (qid, qtext) in enumerate(zip(qids, qtexts)):
        if i == dv_idx:
            continue
        qt = clean_text(qtext)
        ql = qt.lower()
        kscore = sum(1 for k in PERCEPTION_TERMS if k in ql)
        if kscore < 2:
            continue
        profile = response_profile(nonempty_values(data, i))
        if profile["n"] == 0 or is_open_ended(qtext, profile) or not maybe_ordered(profile):
            continue
        if "rank" in ql:
            continue
        if "how likely are you to pursue a cpa license" in ql:
            continue
        overlap = overlap_nonempty(data, dv_idx, i)
        if overlap < 30:
            continue
        perception_candidates.append((kscore, overlap, profile["n"], i, qid, qt))

    # Prioritize directly phrased perception/pathway items.
    perception_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = []
    for item in perception_candidates:
        ql = item[4].lower()
        if any(
            token in ql
            for token in [
                "overall perception",
                "availability",
                "aware",
                "alternative pathway",
                "credit hour",
                "requirements",
                "impacted your desire",
            ]
        ):
            selected.append(item)
        if len(selected) == 3:
            break

    # Fallback: if heuristic filter is too restrictive, keep top overlap-based items.
    if len(selected) < 3:
        selected = perception_candidates[:3]

    # Build modeling rows with complete cases and numeric mappings.
    y = []
    X_cols = [[] for _ in range(2)]  # perception composite + awareness
    kept_rows = 0
    dv_dist = Counter()

    per1_idx = selected[0][3]
    per2_idx = selected[1][3]
    aware_idx = selected[2][3]

    for row in data:
        vals = []
        for idx in [dv_idx, per1_idx, per2_idx, aware_idx]:
            if idx >= len(row):
                vals.append(None)
            else:
                vals.append(row[idx].strip())
        if any(v in (None, "") for v in vals):
            continue

        dv_num = likert_numeric(vals[0])
        p1_num = likert_numeric(vals[1])
        p2_num = likert_numeric(vals[2])
        aware_num = likert_numeric(vals[3])
        if None in (dv_num, p1_num, p2_num, aware_num):
            continue

        perception_index = (p1_num + p2_num) / 2.0
        y.append(float(dv_num))
        X_cols[0].append(float(perception_index))
        X_cols[1].append(float(aware_num))
        dv_dist[vals[0]] += 1
        kept_rows += 1

    beta, r2 = ols(y, X_cols)

    desc = {
        "n_total_responses": len(data),
        "n_model": kept_rows,
        "dv_distribution": dv_dist,
        "dv_mean": mean(y),
        "perception_index_mean": mean(X_cols[0]),
        "aware_share": mean(X_cols[1]),
    }

    details = {
        "dv": {"qid": dv_qid, "question_text": dv_text},
        "perception_predictors": [
            {
                "qid": s[4],
                "question_text": s[5],
                "keyword_score": s[0],
                "overlap_with_dv": s[1],
                "non_missing": s[2],
            }
            for s in selected
        ],
        "coefficients": {
            "intercept": beta[0],
            "perception_index": beta[1],
            "awareness_yes": beta[2],
            "r_squared": r2,
        },
        "descriptives": {
            **desc,
            "dv_distribution": dict(dv_dist),
        },
    }

    os.makedirs(os.path.dirname(DETAILS_FILE), exist_ok=True)
    with open(DETAILS_FILE, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    lines = []
    lines.append("# CPA Pathway Survey: Cross-Sectional Association Analysis")
    lines.append("")
    lines.append("## Question")
    lines.append(
        "How are students’ perceptions of the CPA 150-credit-hour requirement (and alternative-pathway framing) associated with their stated intent to pursue graduate accounting education?"
    )
    lines.append("")
    lines.append("## Data handling and Qualtrics headers")
    lines.append("- Used a parser that treats Qualtrics row 1 as QIDs, row 2 as human-readable question text, and row 3 as import metadata; respondent data starts on row 4.")
    lines.append("- Preserved a QID→question-text mapping in `outputs/analysis_details.json` for interpretability.")
    lines.append("")
    lines.append("## Programmatic variable selection")
    lines.append(f"- **Primary DV selected via header scan:** `{dv_qid}` — {dv_text}")
    lines.append("- Selection rule: scanned question text for intent-related terms (`intent`, `plan`, `graduate`, `MAcc`, `master`, `enroll`, `desire`, `likely`) and retained compact, non-open-ended response items.")
    lines.append("- **Perception predictors selected via header scan** using terms (`150`, `credit`, `hours`, `requirement`, `CPA`, `pathway`, `barrier`, `cost`, `time`, `availability`, `aware`) and keeping direct, non-ranking items:")
    for s in selected:
        lines.append(f"  - `{s[4]}` — {s[5]}")
    lines.append("- Constructed a **perception index** as the mean of aligned Likert scores for the first two selected perception items; included awareness (`Yes`/`No`) as a separate predictor.")
    lines.append("")
    lines.append("## Simple descriptives")
    lines.append(f"- Model sample (complete cases): **{kept_rows}** respondents (out of {len(data)} rows).")
    lines.append(f"- DV mean (1=significantly decreased desire ... 5=significantly increased desire): **{desc['dv_mean']:.2f}**.")
    lines.append(f"- Perception index mean: **{desc['perception_index_mean']:.2f}**.")
    lines.append(f"- Awareness (`Yes`) share: **{desc['aware_share']:.2%}**.")
    lines.append("- DV category counts:")
    for k, v in dv_dist.most_common():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("## Association model")
    lines.append("Because the selected DV is an ordered 5-level intent/desire item, a simple **OLS association model** was fit as a lightweight approximation:")
    lines.append("")
    lines.append("`DV_numeric = b0 + b1*(perception_index) + b2*(aware_before_survey)`")
    lines.append("")
    lines.append(f"- Intercept (b0): **{beta[0]:.3f}**")
    lines.append(f"- Perception index (b1): **{beta[1]:.3f}**")
    lines.append(f"- Awareness yes=1 (b2): **{beta[2]:.3f}**")
    lines.append(f"- R²: **{r2:.3f}**")
    lines.append("")
    dir_p = "positive" if beta[1] > 0 else "negative" if beta[1] < 0 else "near-zero"
    dir_a = "positive" if beta[2] > 0 else "negative" if beta[2] < 0 else "near-zero"
    lines.append(f"Interpretation (association language): in this specification, the perception-index coefficient is {dir_p} (b1={beta[1]:.3f}) and the awareness coefficient is {dir_a} (b2={beta[2]:.3f}); these are cross-sectional associations with self-reported intent.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- This is **cross-sectional** survey data; each respondent is observed once, so results are associational and not causal.")
    lines.append("- The outcome is **self-reported intent/desire**, not observed later enrollment behavior.")
    lines.append("- Item wording and branch logic differ for undergraduate vs. graduate respondents; the selected DV primarily reflects the undergraduate branch.")
    lines.append("- OLS on an ordered Likert DV is a pragmatic simplification for this lightweight report; an ordered-logit specification could be explored in extended work.")

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {REPORT_FILE} and {DETAILS_FILE}")


if __name__ == "__main__":
    main()
