# Is the best backtest just luck?

A "backtest" means trying a trading idea on past prices to see if it would have made money. If you try
enough versions of an idea, one of them will look great by pure luck. This small tool tells you how much of
the best result is probably luck. The longer notes and tables are in `DETAILS.md`.

## Run it (about 5 minutes)

**Easiest: Google Colab, nothing to install**
1. Go to colab.research.google.com and choose **File > Upload notebook**. Pick `PRISM_demo.ipynb` from this folder.
2. Choose **Runtime > Run all**. When the first cell asks for a file, upload the zip of this project.
3. It runs the tests, the demo, and the real NSE test, and shows each result under its cell. About 3 minutes.

**On your own computer** (Python 3.10 or newer)
```
pip install -r requirements.txt
python -m pytest -q                  # 21 tests, expect "21 passed"
python -m strategy_audit demo        # two made-up markets, about 10 seconds; charts go to outputs/
```
Optional, real NSE prices (needs internet, about 30 seconds):
```
python scripts/fetch_nse_eod.py data/nse_eod
python -m strategy_audit real --csv-dir data/nse_eod --placebo 300
```
Optional, the long validation runs (up to about 15 minutes; the saved charts in `outputs/` come from these):
```
python -m strategy_audit calibrate
python -m strategy_audit disclosure
```
**What you should see:** the demo prints one line per market. The market with a hidden signal ends in
`On track`, the one with nothing ends in `At risk`. The real NSE run also ends in `At risk`.
Real-data numbers can shift slightly because the public dataset is updated over time.

## 1. The problem
**What:** PRISM lets people turn an idea ("buy when funds raise their stake") into a test on past prices, and
says it uses "only what was known at the time". Two things can quietly make results look better than they are:
- People try many versions, then show the best one.
- Dates can be wrong. In India, a company files its shareholding report up to 21 days after a quarter ends.
  A test that acts on the quarter-end date is using news nobody had yet.

**Who it helps:** people using a strategy builder, and the team that wants those results to be trusted.
**Why it matters to PRISM:** every strategy result on the platform is only as believable as this check.

## 2. What I built
- A simple test engine. A signal appears after the market closes, the trade happens at the next opening
  price, and trading costs are taken off. It refuses any signal that uses information from the future.
- A luck check. For the 32 versions of an idea, it pretends the same stocks were traded on random days
  instead, many times over, to see how good the "best of 32" looks when nothing real is going on. It subtracts
  that luck, shows how the result fades if you act a few days late, and gives a plain status: On track,
  Mixed or At risk.
- A test of the tool itself, on made-up markets where I know whether a real signal exists.
- A check of the 21-day filing problem, and a run on real NSE prices.

How to run it: see **Run it** at the top. How the numbers are computed: section 2b.

## 2b. How the numbers are computed (step by step)
Worked example: the made-up market with **no** real signal (`python -m strategy_audit demo`).
1. **Find events.** A rule fires when a stock's volume today is more than k times (k = 2, 3, 4 or 5) its median
   volume over the previous 20 days. Combined with holding period (1, 3, 5 or 10 days) and an optional "price
   closed up" filter, that gives 32 versions of the idea.
2. **Price each trade.** Buy at the next day's open, sell at the close `hold` days later. Return = stock return
   minus the average return of all the stocks over the same days, minus 25 basis points of costs. If the
   opening price jumped more than 10% the trade is skipped, as it could not have been done.
3. **Score each version.** Trades that start in the same week are averaged first (they share the market's
   mood), then the weekly averages are tested: z = average / (spread / square root of the number of weeks).
4. **Replay in 200 luck-only worlds.** Keep every stock and the number of events, but move each event to a
   random day of the same stock with similar volatility. Score all 32 versions again in each world, and record
   the best z of each world.
5. **Luck check (family-wise p).** What share of luck-only worlds had a best version at least as good as the
   real best? Example: 0.721, meaning 72% of random worlds do as well.
6. **Winner's-curse correction.** In every luck-only world, pick the best version the same way and note what it
   earned. The average of those is the bias: +26.8 bps. Corrected edge = raw edge minus bias = +5.0 - 26.8 =
   **-21.8 bps**.
7. **Delay test.** Re-run the best version acting 0, 1, 2 ... 10 days late and see how fast the edge fades.
8. **Status.** Corrected edge not positive, or p above 0.20: **At risk**. p up to 0.20: **Mixed**. p up to 0.05
   with positive corrected edge: **On track**. (These cut-offs are my own rule of thumb.)

Same steps on the market with a planted 60 bps signal: best version raw +33.0 bps, bias +27.4, corrected
+5.6 bps, p = 0.015, so **On track**. Code: `engine.py` (steps 1 to 3), `audit.py` (steps 4 to 8).

## 3. What it shows (charts are in `outputs/`)
- `planted_verdict_card.png` and `null_verdict_card.png`: one made-up market with a real signal hidden in
  it (the tool says On track) and one with nothing (At risk).
- `calibration_cost0.png`: with no trading costs, ordinary tests wrongly claim a discovery about 32 times in
  100 when there is nothing there. The tool does it about 3 times in 100. With costs included, every method
  is close to the 5-in-100 target. The tool is also less sensitive, so it can miss weak real signals.
- `disclosure_lag.png`: in my simulation, acting on the quarter-end date made a 10-day trade look about
  26 basis points better than it was, and looked "real" in 55 of 100 markets instead of 23.
- `real_nse_verdict_card.png`: on 48 large NSE stocks from 2021 to 23 Oct 2026, none of the 32 versions of
  a "buy after unusually busy trading" idea made money, even before costs. The tool says At risk.

## 4. What I assumed
- The markets used to test the tool are made up. Costs are 25 basis points per round trip, and the benchmark
  is the average of the same stocks.
- How much prices move around the filing is my guess (100 basis points before it, 30 after), not measured.
  The 21-day rule comes from an exchange compliance calendar, so check it before relying on it.
- "Busy trading" stands in for PRISM's own data (promoter trades, UPI, power demand), which isn't public.

## 5. Limits
- The real prices come from a community dataset on GitHub, not an official source, and only include
  companies that are large today, which flatters results.
- The 32 versions are similar to each other, so they count as fewer separate tries than 32.
- The luck correction is cautious when a real signal exists. The status cut-offs are my own rules of thumb.
- Liquidity, company actions and sector effects are not modelled.

## 6. What I'd improve before real use
1. Use data that records when each number was actually published, including companies that later disappeared.
2. Track how many versions each user really tries over time.
3. Use costs that depend on how easily a stock trades, and compare against similar stocks.
4. Add warnings for stale or thin data, and let people type ideas in plain English.

## 7. What surprised me
- My first check showed the tool raising false alarms 17 times in 100. It was my own bug: one random seed
  was used both to make the market and to pick the random days, so they were linked. After fixing it, the
  rate was 3 to 7 in 100 on 150 fresh markets.
- Once trading costs are included, most ideas simply lose money. The "best of many" problem only really
  shows up when you remove costs.


