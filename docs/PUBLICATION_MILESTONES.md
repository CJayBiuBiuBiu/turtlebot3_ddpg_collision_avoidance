# Publication experiment milestones

## S1. Observed-area control — 2026-11-21

Status: implementation and focused local tests complete. Gazebo pilot pending.

- The default `reference_coverage` mode preserves legacy control behavior.
- The `observed_free_area` mode counts unique laser-cleared grid cells and
  uses area gain in square metres to reset the stall timer. Footprint
  clears and invalid scans do not count. The default 0.5 m² threshold is a
  pilot starting value, not a tuned parameter.
- Observed-area mode requires laser mapping, a finite simulation horizon,
  frozen policy evaluation and disabled coverage-dependent tail and
  adaptive modes. It disables early completion and sends a zero value
  in the legacy policy-observation coverage field. The original geometry
  score stays in the evaluation CSV and displayed progress.
- The observed-area CSV adds the mode and observed free area. Use a new
  output directory and keep original CSVs, checkpoints and manifests.
- Local verification: 22 focused tests pass. Python and launch XML parse.
  The Ubuntu VM reset the SSH connection, so this checkpoint does not
  include new simulation results.

For the paired pilot, keep the world, robot poses, checkpoints, vision
settings and seed fixed. Use `selection_mode:=mappo_hybrid`,
`mapping_mode:=laser_scan`, `max_simulation_seconds:=300`,
`fixed_horizon_only:=true`, `complete_on_coverage:=false` and
`tail_completion_enabled:=false` in both conditions. Switch
`control_progress_mode` between `reference_coverage` and
`observed_free_area`. Supply a distinct `metrics_path` for every run.
Calculate 90% and 97% threshold times from the evaluation CSV.

GitHub checkpoint: review and publish the S1 changes as a separate commit
after reviewing the pre-existing changes in the working tree. Suggested
commit title: `Separate observed exploration progress from reference coverage`.

## S2. Separate vision roles — 2026-11-21

Status: score routing and focused local tests complete. Decision-event
logging and the Gazebo comparison follow separately.

- Use `vision_role:=off|actor_only|rules_only|both` with the same fixed policy
  and environment. The default `both` reproduces the previous scores.
- The rules channel affects ownership and analytic fallback. In
  `rules_only`, both the explicit vision feature and the vision contribution
  to the actor's analytic-cost feature are disabled. The actor can still
  receive a different candidate mask because the rules can change ownership.
  In `actor_only`, vision leaves rule costs and ownership unchanged at the
  same decision state.
- These are interventions on a frozen controller. Changes in performance
  include the effect of changing the input distribution for that controller.
- Local verification: 37 focused tests pass, including numeric equivalence
  to the legacy routing and tests for cross-channel leakage.

GitHub checkpoint: keep S2 separate from S1. Suggested commit title:
`Separate actor and rule-based frontier vision`.

## S3. Trace frontier decisions — 2026-11-21

Pass a new `frontier_trace_path` in observed-area simulation runs to record
the selected frontier, the contemporaneous candidates and their owners,
whether the actor or a rule assigned it, and the terminal outcome. Join
assignments and outcomes by `attempt_id`. Open attempts at the horizon or
shutdown are censored, rather than counted as failures. The listed candidates
describe the coordinator's state at assignment time; an actor message may
come from an earlier observation. The area gain is a team-wide change
over each attempt and must not be summed across overlapping attempts.

GitHub checkpoint title: `S3: Record frontier assignments and outcomes`.
