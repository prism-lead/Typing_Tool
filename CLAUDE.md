# CLAUDE.md — PRISM Survey Engine (Typing Tool)

## Project Overview

The PRISM Survey Engine is a standalone Decipher (Forsta) XML module that:
1. Screens registered U.S. voters via a 7-question screener
2. Assigns respondents to one or two party-specific behavioral models (GOP, DEM, or Both)
3. Administers MaxDiff best-worst scaling exercises (11 items/GOP, 8 items/DEM)
4. Collects attitudinal vectors (DEM model only: justice + industry)
5. Computes real-time PRISM segment assignment using nearest-centroid + softmax

## Repository Structure

```
Typing_Tool/
├── CLAUDE.md                    # This file — project conventions and context
├── README.md                    # Project README
├── generate_bibd.py             # Deliverable 1: BIBD design generation script
├── DemDesign.dat                # Generated DEM MaxDiff design (8 items, 8 tasks, 4/task)
├── GOPDesign.dat                # Generated GOP MaxDiff design (11 items, 11 tasks, 5/task)
├── prism_module.xml             # Deliverable 2: Core Decipher survey XML module
├── prism_styles.xml             # Deliverable 3: Custom CSS/JS styles for Decipher
├── test_harness.py              # Deliverable 4: Validation test harness
└── docs/                        # Reference documentation (if added)
```

## Platform: Decipher / Forsta

- **Survey definition:** Custom XML dialect with embedded Python in `<exec>` blocks
- **Python compatibility:** Python 2/3 compatible (no f-strings; use `format()` or `%`)
- **Documentation:** https://forstasurveys.zendesk.com/hc/en-us
- **MaxDiff:** Uses `setupMaxDiffFile()` and `setupMaxDiffItemsI()` helpers with .dat design files
- **Hidden variables:** Must include `where="execute,survey,report"` for data export visibility
- **Variable prefix:** All computed/hidden variables use `X` prefix convention

## Key Conventions

### Coding Standards
- Python 2/3 compatible syntax in all `<exec>` blocks (no f-strings, no walrus operator)
- Use `format()` or `%` string formatting
- All computation runs in real-time within Decipher `<exec>` blocks — no post-processing
- Module must be self-contained — no host survey dependencies except `standalone_mode` flag

### Variable Naming
- Screener questions: `qvote`, `qzip`, `qage`, `qgender`, `q2024vote`, `qparty1/2/3`, `qparty`
- Computed flags: `TYPING_MODULE`, `standalone_mode`
- GOP outputs: `XGOP_RAW`, `XGOP_SOFTMAX`, `XGOP_SEG_FINAL_1`, `XGOP_SEG_FINAL_2`
- DEM outputs: `XDEM_RAW`, `XDEM_SOFTMAX`, `XDEM_SEG_FINAL_1`, `XDEM_SEG_FINAL_2`
- Cross-model: `XSOFTMAX_ALL`, `XSEG_ASSIGNED`, `XSOFTMAX_TOP`
- Confidence: `XCENTROID_DIST1`, `XCENTROID_DIST2`, `XREL_DIFF`, `XBCS`

### BIBD Design Specifications
- **DEM:** v=8, b=8, k=4, r=4, λ≈1.714 (near-balanced), 20-30 versions
- **GOP:** v=11, b=11, k=5, r=5, λ=2 (perfect BIBD), 20-30 versions
- Output as tab-delimited .dat files with 1-indexed items

### 16 PRISM Segments
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

### Design Tokens
```
Primary font:    Inter
Display font:    Fraunces
Background:      #F5F2ED (cream)
CTA/primary:     #6B7F4E (olive)
Text primary:    #1B2A4A (navy)
Text secondary:  #4A5568 (slate)
Divider/border:  #E2E8F0 (light gray)
Error:           #C53030 (red)
Card radius:     12px
Pill radius:     30px
```

## Build Sequence

1. **Deliverable 1:** `generate_bibd.py` — BIBD design generation → `.dat` files
2. **Deliverable 2:** `prism_module.xml` — Core Decipher survey module XML
3. **Deliverable 3:** `prism_styles.xml` — Custom CSS/JS styles
4. **Deliverable 4:** `test_harness.py` — Validation test harness

## Testing

Run the test harness:
```bash
python test_harness.py
```

Generate BIBD designs:
```bash
python generate_bibd.py
```

Both scripts are Python 3 compatible for local execution; the XML exec blocks maintain Python 2/3 compatibility for Decipher runtime.
