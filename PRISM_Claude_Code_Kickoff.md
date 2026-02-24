# PRISM Survey Engine — Claude Code Build Kickoff

## PROJECT OVERVIEW

Build the PRISM Survey Engine Phase 1: a standalone Decipher (Forsta) XML module that screens registered U.S. voters, assigns them to one or two party-specific behavioral models, administers a MaxDiff best-worst scaling exercise, collects attitudinal vectors (DEM model only), and computes a real-time PRISM segment assignment using a nearest-centroid classifier with softmax probability output.

The complete specification is in the attached **PRISM_Survey_Engine_PRD_v1.2.docx**. Read it fully before writing any code. The attached **PRISM_Master_Specs_for_Typing_Tool.xlsx** contains the authoritative centroid matrices, z-standardization norms, item text, and the current production XML (sheet "Current XML") which demonstrates Decipher idioms for MaxDiff, exec blocks, and routing. The two PNG files (prism_p1.png, prism_p3.png) are the design mockups for the custom UI.

**Platform:** Decipher/Forsta survey platform. Decipher uses a custom XML dialect for survey definition with embedded Python in `<exec>` blocks. Documentation: https://forstasurveys.zendesk.com/hc/en-us — consult this for XML syntax, question types, style system, and MaxDiff implementation patterns.

---

## ATTACHED FILES

1. **PRISM_Survey_Engine_PRD_v1.2.docx** — Full PRD with 14 sections covering architecture, screener spec, model assignment, MaxDiff design, computation engine, design/UX, routing, outputs, testing
2. **PRISM_Master_Specs_for_Typing_Tool.xlsx** — 7 sheets:
   - *Variable Map* — All 54 variables across 4 layers
   - *Master Question Map* — Question text, response options, implementation details, routing
   - *Reduced Typing Tool* — Centroid matrices, z-norms, B-W computation walkthrough (USE THIS for exact centroid values, not the rounded PRD tables)
   - *Current XML* — ~8,600 lines of production Decipher XML showing the existing (more complex) implementation
   - *Design Specs* — Design tokens (fonts, colors, spacing, layout)
   - *HTML Render Tracker* — 17 custom render patterns
   - *Routing & Termination* — 20 routing rules
3. **prism_p1.png** — Design mockup: screener pages (voter reg, ZIP, age, gender, 2024 vote, party ID)
4. **prism_p3.png** — Design mockup: MaxDiff task layout, DEM vector 7-pt scale with color gradient

---

## BUILD SEQUENCE

Work in this order. Complete and test each deliverable before moving to the next.

### Deliverable 1: BIBD Design Generation Script (Python)

Generate balanced incomplete block designs for both MaxDiff exercises.

**DEM Design (near-balanced NBIBD):**
- v=8 items, b=8 tasks, k=4 items/task, r=4 replications/item
- λ = 12/7 ≈ 1.714 (non-integer; pair frequencies will be 1 or 2)
- Each item appears exactly 4 times across 8 tasks
- Generate 20–30 versions

**GOP Design (perfect BIBD):**
- v=11 items, b=11 tasks, k=5 items/task, r=5 replications/item
- λ = 2 exactly (every pair co-appears exactly twice per version)
- Each item appears exactly 5 times across 11 tasks
- Generate 20–30 versions

**Output files:**
- `DemDesign.dat` — Tab-delimited. Columns: version, task, item1, item2, item3, item4. Items are 1-indexed integers.
- `GOPDesign.dat` — Tab-delimited. Columns: version, task, item1, item2, item3, item4, item5. Items are 1-indexed integers.

**The Decipher `setupMaxDiffFile()` function** (visible in the Current XML sheet) reads these files. Match that expected format exactly. Each version is a contiguous block of rows (8 rows for DEM, 11 rows for GOP).

**Validation diagnostics to print:**
- Item frequency per version (each item should appear exactly r times)
- Pair co-occurrence matrix per version (should be λ ± 0 for GOP, λ ± 1 for DEM)
- Aggregate pair balance across all versions
- Confirmation of no duplicate items within any task

---

### Deliverable 2: Module XML (Decipher XML)

The core deliverable. A single contiguous XML block that can be dropped into any Decipher survey.

**Study the Current XML sheet extensively** before writing. It demonstrates:
- The `setupMaxDiffFile()` and `setupMaxDiffItemsI()` helper functions (exec when="init")
- MaxDiff loop construct with quota-based version assignment
- B-W score computation in exec blocks
- Z-standardization with hardcoded norms
- `algorithmCalculation()` and `algorithmRaw()` centroid distance functions
- Segment assignment via min-distance
- Hidden radio/text/float variables for storing computed values
- The `where="execute,survey,report"` pattern for hidden variables

**Module structure (in order):**

```
<!-- Configuration flag -->
standalone_mode = 0  (set to 1 for standalone survey with segment reveal)

<!-- SCREENER (page.group.1) -->
qvote          → radio, atmtable.6; TERM if r2 (No) or r3 (PNTA)
qzip           → number, 5 digit boxes; TERM if 00000
qage           → number, 4 digit boxes; TERM if age<18 or 0000
qgender        → radio, atmtable.6; TERM if r5 (PNTA)
q2024vote      → radio, atmtable.6 with images; TERM if r4 (didn't vote) or r5 (PNTA)
qparty1        → radio, atmtable.6 with images; TERM if r4 (PNTA)
qparty2        → radio, atmtable.6 pill; COND: qparty1=r1 or r2
qparty3        → radio, atmtable.6 with images; COND: qparty1=r3

<!-- COMPUTED: qparty 7-point scale -->
exec block: derive qparty (1=Strong R ... 7=Strong D)

<!-- COMPUTED: TYPING_MODULE assignment -->
exec block: assign TYPING_MODULE = r1 (GOP) / r2 (DEM) / r3 (Both)
  Rules (using q2024vote and qparty):
    Trump + qparty<5         → GOP only
    Trump + qparty>4         → Both
    Harris + qparty<4        → Both
    Harris + qparty>3        → DEM only
    Other + qparty 1-2       → GOP only
    Other + qparty=3 (Lean R)→ Both models
    Other + qparty=4         → TERMINATE (true independent)
    Other + qparty 5-7       → DEM only

<!-- GOP BLOCK (cond: TYPING_MODULE=r1 or r3) -->
MaxDiff: 11 items × 11 tasks × 5 items/task
  - exec when="init": load GOPDesign.dat via setupMaxDiffFile()
  - Quota-based version assignment
  - Loop construct with radio (adim="cols", grouping="cols", shuffle="rows")
  - B-W score computation → hidden text grid
  - Z-standardization (11 norms, see below)
  - Centroid distance computation (11 dims × 10 segments)
  - Softmax: P(seg_k) = exp(-D²_k) / Σ exp(-D²_j) for k=1..10
  - Store: XGOP_RAW (distances), XGOP_SOFTMAX (10 probabilities),
    XGOP_SEG_FINAL_1 (nearest), XGOP_SEG_FINAL_2 (runner-up)

<!-- DEM BLOCK (cond: TYPING_MODULE=r2 or r3) -->
MaxDiff: 8 items × 8 tasks × 4 items/task
  - Same pattern as GOP block but with DemDesign.dat
  - B-W score computation → hidden text grid (8 items)

<!-- DEM VECTORS (page.group.4, cond: TYPING_MODULE=r2 or r3) -->
vector_justice   → 7-pt agree/disagree (1=Strongly disagree ... 7=Strongly agree)
vector_industry  → 7-pt agree/disagree, REVERSE CODED: compute (8 - raw) before z-scoring
  - Display: color gradient on hover/select states across 7 points

<!-- DEM CENTROID COMPUTATION -->
  - Append z-scored vectors as dims 9-10 of the 10-element input vector
  - Centroid distance (10 dims × 6 segments)
  - Softmax: P(seg_k) = exp(-D²_k) / Σ exp(-D²_j) for k=1..6
  - Store: XDEM_RAW, XDEM_SOFTMAX, XDEM_SEG_FINAL_1, XDEM_SEG_FINAL_2

<!-- XSEG_ASSIGNED RESOLUTION -->
  - XSOFTMAX_ALL: text grid r1-r16 × c1, stores all 16 softmax probabilities
    GOP segments → r1-r10, DEM segments → r11-r16
    Single-model respondents get zeros for the model they didn't take
  - XSEG_ASSIGNED = segment with highest softmax across all models taken
  - XSOFTMAX_TOP = the winning softmax probability

<!-- CONFIDENCE METRICS -->
  - XCENTROID_DIST1 = min distance
  - XCENTROID_DIST2 = second-min distance
  - XREL_DIFF = (DIST2 - DIST1) / DIST1
  - XBCS = High/Medium/Low confidence flag (0.2 threshold on XREL_DIFF)

<!-- SEGMENT REVEAL (cond: standalone_mode=1) -->
  - HTML page displaying assigned segment name
```

**Critical implementation details:**

**GOP Z-Standardization Norms (μ, σ):**
```
ZBW_1  (r1  marriage/family):     μ=0.38509,   σ=1.43382
ZBW_2  (r2  pharma distrust):     μ=-0.17378,  σ=1.33589
ZBW_3  (r3  drug/hospital prices):μ=0.24627,   σ=1.27153
ZBW_4  (r4  rural hospitals):     μ=-0.04010,  σ=0.99507
ZBW_5  (r5  vaccine skepticism):  μ=-0.59229,  σ=1.43007
ZBW_7  (r7  holistic medicine):   μ=-0.02108,  σ=1.03988
ZBW_8  (r8  whole-food living):   μ=0.47455,   σ=1.17742
ZBW_9  (r9  carnivore/low-carb):  μ=-0.37738,  σ=1.06212
ZBW_10 (r10 US supply chain):     μ=0.74447,   σ=1.19670
ZBW_11 (r11 free market+FDA):     μ=-0.40308,  σ=1.23144
ZBW_12 (r12 competition/dereg):   μ=0.53265,   σ=1.19631
```

**DEM Z-Standardization Norms (μ, σ):**
```
ZBW_1  (r1  tech/data equity):    μ=-0.4545,   σ=1.01042
ZBW_2  (r2  mainstream experts):  μ=-0.2546,   σ=1.19910
ZBW_3  (r3  digital disruption):  μ=-0.9328,   σ=1.22412
ZBW_4  (r4  corporate greed):     μ=0.0386,    σ=1.13192
ZBW_5  (r5  Medicare for All):    μ=0.9975,    σ=1.32821
ZBW_6  (r6  union protection):    μ=-0.1851,   σ=1.00078
ZBW_7  (r7  racial equity):       μ=0.1674,    σ=1.09544
ZBW_9  (r9  faith-based access):  μ=-0.6671,   σ=1.51346
ZFACTOR_JUSTICE  (vector):        μ=5.009,     σ=1.28372
ZFACTOR_INDUSTRY (vector,reversed):μ=5.631,    σ=1.02132
```

**Centroid matrices:** Pull exact values from the "Reduced Typing Tool" sheet in the xlsx, NOT from the PRD tables (which are rounded for readability). The GOP matrix is 11 dimensions × 10 segments. The DEM matrix is 10 dimensions × 6 segments.

**16 PRISM Segments (canonical names and order for XSEG_ASSIGNED r1-r16):**
```
GOP (r1-r10):
  r1  Consumer Empowerment Champions (CEC)
  r2  Holistic Health Naturalists (HHN)
  r3  Traditional Conservatives (TC)
  r4  Paleo Freedom Fighters (PFF)
  r5  Price Populists (PP)
  r6  Wellness Evangelists (WE)
  r7  Health Futurists (HF)
  r8  Vaccine Skeptics (VS)
  r9  Medical Freedom Libertarians (MFL)
  r10 Trust The Science Pragmatists (TSP)

DEM (r11-r16):
  r11 Universal Care Progressives (UCP)
  r12 Faith & Justice Progressives (FJP)
  r13 Health Care Protectionists (HCP)
  r14 Health Abundance Democrats (HAD)
  r15 Health Care Incrementalists (HCI)
  r16 Global Health Institutionalists (GHI)
```

---

### Deliverable 3: Custom CSS/JS Styles (Decipher Style XML)

Match the design mockups (prism_p1.png, prism_p3.png) as closely as possible while ensuring data and operational integrity in Decipher.

**Design tokens:**
```
Primary font (body):    Inter
Display font (headers): Fraunces
Background:             #F5F2ED (cream)
CTA / primary action:   #6B7F4E (olive)
Text primary:           #1B2A4A (navy)
Text secondary:         #4A5568 (slate)
Divider / border:       #E2E8F0 (light gray)
Error state:            #C53030 (red)
Card radius:            12px
Pill radius:            30px
```

**Required render patterns for Phase 1:**
1. **atmtable.6 round** — Full-width rounded buttons (12px radius, olive active state). Used by: qvote, qgender, qparty1, qparty3
2. **atmtable.6 pill** — Pill-shaped buttons (30px radius). Used by: qparty2
3. **atmtable.6 with images** — Buttons with left-aligned candidate/party images. Used by: q2024vote, qparty1, qparty3
4. **atmtable.6 with dividers** — Row dividers between groups. Used by: q2024vote
5. **Digit box** — Individual input boxes per digit. Used by: qzip (5 boxes), qage (4 boxes)
6. **MaxDiff grid** — MOST/LEAST two-column layout with navy header bar, bordered cells, centered radios. Used by: DEM and GOP MaxDiff tasks
7. **7-pt semantic differential with color gradient** — Agree/disagree scale, endpoint anchors only, color gradient on hover/select states transitioning across the 7 points. Used by: vector_justice, vector_industry
8. **Segment reveal card** — Segment name display with model-specific branding (standalone mode only)

All styles must be scoped to PRISM-specific class names to avoid conflicts when the module is dropped into host surveys.

Refer to the Design Specs and HTML Render Tracker sheets in the xlsx for detailed token values and pattern specifications.

---

### Deliverable 4: Test Harness (Python)

A validation script that feeds synthetic test vectors through the computation pipeline and verifies correct segment assignment.

**Test cases to include:**
1. **16 prototypical respondents** — One per segment, using B-W scores and vector values that should produce a clear assignment to that segment (use centroid values as the input vector — these should assign to themselves with distance ≈ 0)
2. **Boundary cases** — Input vectors positioned near the midpoint between the two closest centroids for each model, verifying that the correct segment wins and that XREL_DIFF is low
3. **Extreme/uniform cases** — All-zero B-W scores, all-max B-W scores, uniform scores
4. **Dual-model resolution** — Test cases where a respondent takes both models; verify the highest softmax across all 16 segments wins
5. **Routing verification** — All 9 combinations of q2024vote × qparty that affect TYPING_MODULE assignment
6. **Termination verification** — All 9 termination conditions trigger correctly
7. **vector_industry reverse coding** — Verify (8 - raw) is applied before z-scoring

**Output:** Pass/fail for each test case with expected vs. actual segment assignment, softmax probability, and confidence metrics.

---

## KEY DECIPHER PATTERNS TO FOLLOW

Study these patterns in the Current XML sheet before implementing:

1. **MaxDiff via indices method:** `setupMaxDiffFile("DemDesign.dat")` loads the design. `setupMaxDiffItemsI()` maps items to question rows per version/task. Version assigned via quota sheet.

2. **Hidden computed variables:** Use `<radio label="XVAR" optional="1" where="execute,survey,report">` for categorical outputs, `<float label="XVAR" where="execute,survey,report">` for continuous, `<text label="XVAR" where="execute,survey,report">` for grids.

3. **Exec blocks for computation:** All centroid math runs in `<exec>` Python blocks. The current XML's `algorithmCalculation()` and `algorithmRaw()` functions are a clean pattern to follow.

4. **Softmax addition (new for this build):** The current XML uses a logit model as a secondary classifier. This build replaces that with softmax on centroid distances:
```python
import math
distances = [float(x) for x in XGOP_RAW.values]
neg_dists = [-d for d in distances]
max_neg = max(neg_dists)  # numerical stability
exp_vals = [math.exp(d - max_neg) for d in neg_dists]
total = sum(exp_vals)
softmax = [e / total for e in exp_vals]
```

5. **Termination:** Use `<term label="Term_LABEL" cond="CONDITION">Message</term>` elements.

6. **Conditional display:** Use `cond` attribute on blocks and questions: `<block label="DEMBlock" cond="TYPING_MODULE.r2 or TYPING_MODULE.r3">`

---

## IMPORTANT CONSTRAINTS

- All computation must execute in real-time within Decipher `<exec>` blocks during the live survey session. No post-processing or external computation.
- The `<exec>` environment is Python 2-compatible in many Decipher instances. Use Python 2/3 compatible syntax (avoid f-strings, use `format()` or `%` formatting).
- Every hidden variable must include `where="execute,survey,report"` to appear in data exports.
- The module must be self-contained — no dependencies on host survey variables except the `standalone_mode` flag.
- Use the `X` prefix convention for all computed/hidden variables.
- Centroid values and z-norms are hardcoded in exec blocks (not loaded from external files).
- BIBD design files ARE loaded from external .dat files in the Decipher file library.
