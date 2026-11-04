"""Player-aware shooting skill (Master V2 research track).

Modules:
  taxonomy     skill levels / categories (general, type, location, action, location x action, contest)
  history      historical shots with out-of-fold shot-difficulty residuals (context-adjusted outcomes)
  hierarchy    empirical-Bayes hierarchical shrinkage with parent fallback, n_eff and reliability
  glmm         penalized hierarchical logistic adjustment on top of a base model (offset)
  contest      PlayerDashPtShots vs Blueprint contest-distance audit and contest sensitivity
  physical     roster heights (when legitimately available) and data-driven shot-profile archetypes
  features     as-of-date per-shot player-skill features for the master_v2 dataset
  experiments  architecture comparison on the frozen Master V1 split
  audit        diagnostics of the Master V1 player features
"""
