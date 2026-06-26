# Notes

Intended both as a personal reference and as onboarding material for new joiners.

---

## 1. Team Footprint & Scope

- A Director manages the book; the whole team trades on the same book. PnL is measured at book level, not per individual trader.
- Books
  - **Flow** — cap/floor/swaption, market-making OTC, taker on listed. focuses on 1Y–5Y swaption vol, where both OTC and listed markets exist and a basis must be managed.
  - **EXO** — model owners; provide pricing, marking tools and onboard new products.
  - **CMS Spread** — historically required by risk to be ring-fenced from the rest of the Exo book; in practice the trading view is still consolidated with the rest of Exo.
  - **EXO** — xx
  - **HYB** — xx


---

## 2. Workflow & Platforms

### 2.1 Trade lifecycle
- Most trades originate from **voice / chat negotiation** (chats are the dominant inbound channel; a flow trader typically processes ~50 tickets per half-day).
- Trader does theoretical pricing, then adds spread based on the **market context and current book** (this is the hardest step).
- Trades are confirmed via platforms — **Markitwire** is for operationalization / confirmation only; it does **not** participate in price discovery. Once both sides have negotiated, the trade is entered and the counterparty matches it.

### 2.2 Trading channels
| Product | Channel |
|---|---|
| Swaps | Tradeweb |
| Futures | Fits |
| Listed | Bloomberg (BBTI) |
| Swaptions | Markitwire |
| Structured products | SG Markets |

### 2.3 Booking rules
- **SG Markets / Markitwire** → auto-booked.
- **Fits** → auto-booked in US, **manual** in APAC.
- **Email / chat trades** → typically booked manually by **TSU**.
- **Internal trades** → booked by the desk itself.

### 2.4 Internal tools
- **Highway** — risk computation platform.
- **FAR** — data store.
- **XOne** — core pricing / booking / lifecycle system.


---

## 3. Key Market Terminology

- **TN / SN / SW** — Tom-Next, Spot-Next, Spot-Week. TN and SN are overnight tenors; SW is one week.
- **CMT (Constant Maturity Treasury)** — e.g. 10Y CMT is the theoretical 10Y yield interpolated off the current Treasury curve, *not* the on-the-run yield.
- **GC (Generalised Collateral)** — anything that qualifies as collateral. In rates this is mainly on-the-run UST and agency paper (e.g. agencies / GSEs); in Equity Repo it extends to single names (e.g. AAPL).
- **NnM (New and Modified)** — bucket used in the **daily PnL** decomposition.
- **Fugit** — expected (not contractual) maturity. Required for callable products; usually estimated via Monte Carlo because the effective life of a 5Y callable is shorter than 5Y.
- **IPV (Independent Price Verification)** — a separate marking policy aligned to **external market parameters** (other banks' marks), used by risk / compliance. External parameters are sourced by the **Financing Department**.
- **Totem** — monthly (or daily for liquid points) bid/ask survey. Not a tradable venue. Submitters get to see (anonymised, with quantile distribution) what the rest of the street is submitting. The output is used to **calibrate vol**, particularly for less liquid points.

---

## 4. Margins, Fees, Booking

- **X = Sales margin**, **Y = Tech (trader) margin**. Sales takes X, trader takes Y. Sometimes an **UF (upfront)** is paid to an intermediary.
- **Sales Credit:** `SC = X + F(Y)` where `F(Y)` is the **tech release / tech reco** (e.g. tech reco at 50% means SC = X + 0.5·Y).
- **Funding Boost** — internal preferential funding. Normal funding pays SOFR + spread (e.g. SOFR+60bps) periodically; FB capitalises the discount and pays it **upfront** as a one-shot.
- **Minimum margin policy:** `min margin ≥ 20bps × fugit`. For a 5Y callable the relevant fugit is the MC-estimated effective life, not 5Y.

### 4.1 Structured-note cashflows (internal mechanics)
When the desk sells a structured note:
1. Client pays a large notional and receives `coupon1` periodically from the desk.
2. The notional is transferred **internally to Financing**; the desk receives `coupon2` from Financing periodically.
3. The desk's actual cashflow is **+coupon2 / −coupon1**.

The desk monitors daily whether the **PV received** is still **≥ PV paid**; if it flips, the trade is **called**.

---

## 5. Funding & Hedging Workflow

### 5.1 Monday Funding
- Run weekly on Monday with the **Treasury desk** to hedge funding positions.
- Each side reports a need (notional, tenor, direction). If an internal match is found the cross is at **mid**; otherwise the desk pays the Treasury desk a spread.

### 5.2 Delta hedging — term partition
Delta is run **partitioned by tenor** and each bucket has to be neutral. Each bucket uses different hedging instruments depending on liquidity:
- **Short end (≤ 6M)** → **futures** (deepest liquidity).
- **Mid (~2Y)** → mixed.
- **Long end (2–5Y, 5–10Y, …)** → **swaps** (better long-dated liquidity).

### 5.3 Greeks — units & conventions
- All P/L and risk reported in **EUR**.
- **Vega** — kEUR per daily-vol bp.
- **Delta** — kEUR per 10bps shift.
- **Gamma** — no single scalar; read directly off the delta ladder (ATM, ATM−2bps, ATM−5bps, …).
- For Exo, basic Greeks (delta, gamma, vega) are tracked continuously; higher-order exposures are managed by experience.
- For Flow, **Bond option / bond-future option / swaption** — their **deltas are intrinsically different**. Look at them separately *and* aggregated.

---

## 6. Vol Surface — Working Practices

- The desk regularly looks at vol points **far out-of-the-money** (e.g. ATM+5σ). Reason: CMS replication.

### 6.1 Why far-OTM vol matters — Hagan CMS replication
The Hagan static replication formula for a CMS payoff under the annuity numeraire:
$$V_{CMS} = h(S_0)\,A_0 + \int_0^{S_0} h''(K)\,\text{Rec}(K)\,dK + \int_{S_0}^{\infty} h''(K)\,\text{Pay}(K)\,dK$$
where $h(S)$ is the CMS payoff expressed under the annuity numeraire. $h$ is **monotonically increasing and convex**, so the integrand puts material weight on **high-strike payers** — meaning the **high-rate (far-OTM) vol regime is amplified** in a CMS price. That is why traders care about +5σ marks.

### 6.2 Vol-addon (Flow only)
- Only relevant where OTC and listed coexist (1Y–5Y swaption space).
- The **listed price is typically off**; a vol-addon is added to OTC vol so the OTC / listed comparison is apples-to-apples.


### 6.3 Calibration
- Calibration is fundamentally an **optimisation problem**: fit the market and stay smooth.
- In illiquid regions the quant model can drift (e.g. the ticker the model references doesn't reflect real pricing, or a more meaningful ticker is missing from the model). The **trader can override** with a manual mark and re-calibrate.

### 6.4 Strength on the surface
Every bank has a region of the **expiry × tenor** surface where it can show competitive prices; this is driven by:
1. **Inventory.**
2. **Client mix** (knowing what they want to trade).
3. **QIS regular rebalancing flows** (the desk knows these in advance).

Current desk strength: **1Y–10Y expiry × 10Y–30Y tenor**.

---

## 7. Product Catalogue

### 7.1 Callable notes (80%+ of the exo trades are callable)
- The defining feature of most of the desk's structured note flow.
- Fugit and min-margin rule (§4) drive the marking.

### 7.2 Phoenix
Two barriers and a set of observation dates:
- **Coupon barrier** — if rate is below this on an observation date, coupon `c%` is paid. Example: spot 3.9%, coupon barrier 4.4% → coupon as long as obs < 4.4%.
- **Autocall barrier** — if rate is below this, the trade **terminates early**. Example: 3.4%.
- **Memory feature:** if obs are 4.5 / 4.5 / 4.3 — without memory only the last period pays one coupon; **with memory, the final period pays all three accrued coupons**.

### 7.3 Commercial Period (pre-sale window)
- The product is "pre-sold" at T with a commercial period (e.g. 4 months). Clients can fund gradually or in one shot by T+4M; the trade officially starts at T+4M.
- **Risk:** the desk has to start hedging *during* the commercial period. If rates move sharply and the client backs out, the desk is left with an unwanted hedge — material loss.
- **Floating-strike / floating-barrier** designs exist to mitigate this but are still rare.

### 7.4 Hybrids
A base rates product overlaid with one additional risk dimension (typically **credit** or **FX**) — more risk, fatter coupon.
- **Defaultable note** — links to a third-party reference entity's default. Equivalent to selling CDS on top of the base structure; coupon is enhanced by the CDS premium.
- **FX hybrid** — base note plus an FX-linked structure (e.g. an FX knock-out).
- **Client base** — primarily Swiss and Singapore **private banks**.
- Because the hybrid book is **small**, the desk only hedges **first-order** exposure in the extra dimension. FX-rate / correlation cross-exposures are usually left unhedged — covered instead through higher fees.

### 7.5 Repackage bonds (Repacks)
- Used when a client wants to **isolate themselves from the bank's credit risk**.
- An **SPV** (bankruptcy-remote wrapper) holds the bonds. The SPV is a standalone entity, not the bank's balance sheet.

### 7.6 FIC QIS VRR (Vol Risk Premium) — desk's largest QIS exposure
- **QIS replication is mechanical** — traders hedge strictly to the white paper.
- **QIS clients** are mostly large asset managers and pension funds (small teams running large AUM that need systematic, low-maintenance products). 
- **Capacity matters** — too much size in a QIS raises hedging costs and creates execution-mechanism leakage. 
  
Best-selling FIC QIS strategy. Mechanics:

- Long vol is usually a **negative-carry** position via the **IV term-structure rolldown** — but at **long tenors this flips**: long-tenor long-vol benefits from **roll-up**.
- Structural inequality: **forward vol < spot vol**.
- **Rates tend to trend on horizons > 6M.**
- Resulting playbook: in **long, liquid expiry buckets**, find the expiry with the **highest `spot vol / 1Y-fwd vol` ratio** → **buy the forward (cheap vol), sell the spot (rich vol)**. Delta-hedge with a trend-aware adjustment.

### 7.7 QIS pricing structure (operational)
- Sell price = **mid + predetermined cost**. Hedging costs are embedded in the fixed cost.
- The index is calculated **both** by the bank and by an external **agent (usually Bloomberg)**; the index is publicly published on BBG.
- Client P/L example: actual product PnL goes 1 → 1.11, but the **index goes only to 1.10** — the 1bp difference is the embedded fee.
- QIS has a **predetermined maturity** (6M / 1Y / 2Y …) — never open-ended fund-style. Payoff styles:
  - **Bullet** — single payout at maturity.
  - **Semi-cliquet** — payouts in tranches.
- **No price negotiation with clients** — the fee is fixed in the white paper. Clients can shop the white paper to other banks, but replication takes months and burning a relationship that way is reputation-ending — in practice it doesn't happen.

---

## 8. Risk Management & PnL

### 8.1 Exo risk
- Exo risk profile is **too broad to track every dimension** — delta / gamma / vega are tracked continuously; the rest by experience.
- Exo generally **does not look into the underlying model data** (too complex); deep model work is owned by ARD.

### 8.2 PnL match (with independent Risk)
- Risk re-computes the desk's PnL **independently** from the book and reconciles to the desk's own decomposition. They mostly match.
- "Correct" PnL explanation is **subjective** — e.g. OTC swaption has no clean tape, so what counts as the reference price is itself a judgement call.

### 8.3 New product onboarding
- New products require a **FOPI** (Front Office Pricing Information) document — a notebook-style write-up of the risk profile.
- "New product" includes incremental combos: e.g. if RCN exists in the library and the desk wants to trade RCN Autocall, that needs a FOPI.
- Implementation is typically staged: first VBA (calling base products + manually-coded MC for knock-in/out), then C# only once volume justifies it.

### 8.4 ARD / desk interaction
- ARD support work is mostly **pricing and marking** — risk reports and market-data plumbing are owned elsewhere.
- Typical ARD day-to-day for non-linear rates: marking, historical-data research, building small tools (e.g. QIS toolkits), new-product risk review.
