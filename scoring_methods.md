# AI Eligibility Scoring Engine Methods

This document explains the scoring mechanism and formulas used by the **AI Eligibility Scoring Engine**, along with justification structures and diverse categorization techniques to present accurate models to clients.

## 1. Updated Farmer Attributes
The model accepts expanded attributes capturing demographic, geographic, and socio-economic information:
* `state`, `district`, `village`
* `land_size` (total cultivable land in hectares)
* `crop_type` (e.g. Wheat, Vegetables, Spices)
* `irrigation_type` (e.g. Rainfed, Tube well)
* `annual_income`
* `cast_category` (e.g. SC, ST, OBC, General) 
* `age`, `gender`, `created_at`, `updated_at`

## 2. Dynamic Categories for Diversification
To prevent displaying uniform recommendations (e.g., all top 5 choices simply being local tractor subsides), schemes are intelligently mapped into categorical buckets using NLP heuristics on name and benefits tags:
1. **Crop Insurance:** (Keywords: bima, insurance)
2. **Irrigation Support:** (Keywords: sinchai, irrigation, pump, drip, sprinkler)
3. **Farm Equipment & Mechanization:** (Keywords: tractor, mechanization, equipment)
4. **Crop & Infrastructure:** (Keywords: seed, orchard, greenhouse, tissue culture)
5. **Health & Soil Management:** (Keywords: soil health, plant health, ayush)
6. **Direct Financial Support:** (Keywords: samman nidhi, pension, financial)

The engine yields the **highest-ranked schemes per category** (e.g., top 1-2 per bucket), assembling an elite, diversified portfolio of top 5 recommendations.

## 3. The Objective Scoring Funnel
We evaluate candidates by tallying a `Relevance_Score` (demographics) and `Priority_Score` (value magnitude), outputting explicit, detailed **justifications** for each boolean check. 

### A. Non-Negotiable Filter (Immediate `Score = 0`)
* **State Limitation:** Evaluates to 0 if the scheme is explicitly mapped to an alternate State and is not Nationwide.
* **Land Bounds Deviation:** Evaluates to 0 if scheme dictates "small and marginal" ($\le$ 2.0 ha), but farmer has $>$ 2.0 ha.
* **Age Out-of-Bounds:** Evaluates to 0 if age strictly mandates bounded intervals not fulfilled by the user. 
* **Income Ceiling Violation:** Evaluates to 0 if explicit references limit aid solely to low-income brackets below the farmer's stated income.

*In all disqualification cases, a strict string stating `"Ineligible: [Reason]"` is appended for transparent audit trails.*

### B. Relevance Scoring (`$R$`)
Applies positive reinforcements for exact matching text segments against traits:
* `+30`: State Match / Applicability Match
* `+20`: Fulfilling Land Size targets (landless vs $\le$ 2.0 ha targets).
* `+15`: Belonging to targeted vulnerable segments (SC/ST/OBC, Women).
* `+15`: Crop specificity align (e.g., Spice subsidies to Spice/Horticulture farmers).
* `+15`: Appropriate intervention targets (e.g., Drip Irrigation to historically Rainfed lands). 

### C. Priority Scoring (`$P$`)
Surfaces robust structural nets:
* `+10`: Fast liquid transfer / Subsidy grants.
* `+15`: Annuity structures / Pension lines.
* `+15`: Bima offsets / Insurance loss shields.
* `+10`: High Volume financial brackets. 

### Output Result 
The final object renders `Total_Score(S)`, `Category(S)`, and a string-list `Reasons(S)` outlining the point deductions and additions to explicitly justify to the farmer entirely why the scheme has surfaced.
