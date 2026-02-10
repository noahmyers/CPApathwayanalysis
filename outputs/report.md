# CPA Pathway Survey: Cross-Sectional Association Analysis

## Question
How are students’ perceptions of the CPA 150-credit-hour requirement (and alternative-pathway framing) associated with their stated intent to pursue graduate accounting education?

## Data handling and Qualtrics headers
- Used a parser that treats Qualtrics row 1 as QIDs, row 2 as human-readable question text, and row 3 as import metadata; respondent data starts on row 4.
- Preserved a QID→question-text mapping in `outputs/analysis_details.json` for interpretability.

## Programmatic variable selection
- **Primary DV selected via header scan:** `Q52` — How has the availability of (or knowledge about) the alternative pathway to CPA licensure impacted your desire to pursue a graduate degree (MAcc or MBA)?
- Selection rule: scanned question text for intent-related terms (`intent`, `plan`, `graduate`, `MAcc`, `master`, `enroll`, `desire`, `likely`) and retained compact, non-open-ended response items.
- **Perception predictors selected via header scan** using terms (`150`, `credit`, `hours`, `requirement`, `CPA`, `pathway`, `barrier`, `cost`, `time`, `availability`, `aware`) and keeping direct, non-ranking items:
  - `Q6` — What is your overall perception of the change to CPA licensure pathways that creates an alternative pathway with fewer credit hours and an extra year of work experience?
  - `Q51` — How has the availability of (or knowledge about) the alternative pathway to CPA licensure impacted your desire to pursue the CPA license?
  - `Q53` — Were you aware of the alternative pathway to CPA licensure before taking the survey?
- Constructed a **perception index** as the mean of aligned Likert scores for the first two selected perception items; included awareness (`Yes`/`No`) as a separate predictor.

## Simple descriptives
- Model sample (complete cases): **145** respondents (out of 206 rows).
- DV mean (1=significantly decreased desire ... 5=significantly increased desire): **2.86**.
- Perception index mean: **3.85**.
- Awareness (`Yes`) share: **60.69%**.
- DV category counts:
  - No change in desire: 64
  - Decreased desire: 33
  - Increased desire: 31
  - Significantly decreased desire: 13
  - Significantly increased desire: 4

## Association model
Because the selected DV is an ordered 5-level intent/desire item, a simple **OLS association model** was fit as a lightweight approximation:

`DV_numeric = b0 + b1*(perception_index) + b2*(aware_before_survey)`

- Intercept (b0): **3.328**
- Perception index (b1): **0.006**
- Awareness yes=1 (b2): **-0.807**
- R²: **0.174**

Interpretation (association language): in this specification, the perception-index coefficient is positive (b1=0.006) and the awareness coefficient is negative (b2=-0.807); these are cross-sectional associations with self-reported intent.

## Limitations
- This is **cross-sectional** survey data; each respondent is observed once, so results are associational and not causal.
- The outcome is **self-reported intent/desire**, not observed later enrollment behavior.
- Item wording and branch logic differ for undergraduate vs. graduate respondents; the selected DV primarily reflects the undergraduate branch.
- OLS on an ordered Likert DV is a pragmatic simplification for this lightweight report; an ordered-logit specification could be explored in extended work.
