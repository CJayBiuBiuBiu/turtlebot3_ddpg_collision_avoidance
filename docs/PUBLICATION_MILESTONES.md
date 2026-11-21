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
