# Project Memory

## Purpose

This repository is intended to collect and organize materials related to CTA, deep learning, market timing, regime detection, momentum, and systematic portfolio research.

## Current State

- The project currently contains a `papers/` directory with research PDFs.
- The repository is intended to be synchronized through GitHub at `https://github.com/StephBay30/dl-for-cta.git`.
- Persistent assistant/project context is stored in this file.
- Cross-platform line endings are controlled by `.gitattributes`.
- In this Windows/Codex environment, `.git` may be blocked by ACL sandboxing; `.gitdata/` can be used as a local fallback git metadata directory if needed.

## Working Rules

- Keep important project context in `agent_context/memory.md`.
- Keep assistant-facing maintenance instructions in `agent_context/AGENTS.md`.
- Update this memory file when important project decisions, TODOs, milestones, or setup details change.
- Do not write secrets, passwords, API keys, tokens, or private credentials into this repository.

## Decisions

- Use GitHub as the source of truth for syncing the project across machines.
- Store project memory in Markdown files inside `agent_context/`.
- Use LF line endings for text files so the repository behaves consistently on Windows and Linux.
- Ignore local git metadata directories via `.gitignore`.
- Implement the A-share index minute framework directly, not a daily futures v1.
- Use TOML as the experiment parameter interface and expose CLI commands through `python -m dl_for_cta.cli`.
- Implement CPD with the paper's Matérn 3/2 GP versus sigmoid changepoint GP, not a statistical proxy.
- Use a strict train/validation/test workflow: train on the configured train range, select the best
  epoch and TOML grid-search candidate by validation after-cost Sharpe, then run test once from
  `first_test_start` using `best_model.pt`.
- Long-running CLI stages must use structured logging for start/finish, row counts, CPD progress,
  training epoch metrics, validation Sharpe, selected checkpoints, and backtest outputs.

## TODO

- Finish initializing the local git repository and push it to GitHub.
- Consider adding a README that explains the project purpose and paper organization.
- Consider adding metadata for papers, such as title, topic, source, year, and notes.
- Build out performance optimizations for GP-CPD, including better parallelism and shard-level scheduling.
- Add stronger baseline strategies and expanding-window experiment orchestration.
- Extend the current single train/valid/test split into true multi-window expanding evaluation.

## Notes

- A previous attempt initialized `.git`, but `git add .` was blocked by a Windows permission issue on `.git/index.lock`.
- Slow Momentum with Fast Reversion core idea: add an online Gaussian-process changepoint detection
  feature module to a Deep Momentum Network. CPD fits Matern 3/2 GP vs changepoint-kernel GP on
  rolling returns windows, outputting changepoint severity `cp_score` and normalized location/run-length
  `cp_rl_{LBW}` into the LSTM. LSTM sequence length is 63, target return is volatility-scaled next-day
  return, objective is negative annualized Sharpe. Paper reports best CPD LBWs around 21/63 days and
  optimized LBW Sharpe about 2.16 vs LSTM 1.62 on raw signal, before transaction costs.
- A-share framework: use only continuous index data first, excluding index futures because
  futures contracts are discontinuous. Default symbols are `000016.XSHG`, `000300.XSHG`,
  `000905.XSHG`, and `000852.XSHG`. Minute data lives in
  `E:\quant\lyquant\short_arb_firm\data\min_bar`; system Python has `pyarrow`, `torch`, `scipy`,
  `pandas`, and `numpy`.
- Minute strategy design: model emits one target position per minute per index, each in `[-1, 1]`.
  Each index is predicted independently and the portfolio aggregates indices equally. Positions may
  be held overnight. Signal at minute `t` trades at the next minute open to avoid current-bar lookahead.
  Default transaction cost is single-side 5bp, configurable.
- Threshold/no-trade band design: support fixed configurable thresholds first, with grid search such
  as `0`, `0.02`, `0.05`, `0.10`, `0.20`; later add a trainable no-trade band using soft turnover
  penalty during training and a hard threshold during backtest.
- Minute model targets and features: support configurable single or multiple horizons such as `[1]`,
  `[5]`, or `[1, 5, 30]`. Features should include minute returns over `1/5/15/30/60/120/240`, short-term
  reversal, momentum gaps, rolling volatility, intraday range, volume/turnover z-scores, time-of-day
  features, and CPD features. Initial CPD windows should be `60`, `240`, and `1200` minutes, outputting
  `cp_score_{window}` and `cp_loc_{window}`.
- Framework outputs and safety rules: include baselines `long-only`, minute momentum, minute reversal,
  and MACD, plus models `LSTM`, `LSTM+CPD`, and `LSTM+CPD+threshold`. Reports should include annual
  return, volatility, Sharpe, max drawdown, turnover, trade count, and cost-before/after metrics. All
  features must use only minute `t` and earlier; positions generated at `t` apply only from `t+1`.
