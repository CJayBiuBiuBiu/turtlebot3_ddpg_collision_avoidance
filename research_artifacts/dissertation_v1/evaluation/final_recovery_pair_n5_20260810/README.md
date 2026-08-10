# Final-stack recovery ablation

This directory contains a five-seed, one-factor Gazebo comparison between the
final `mappo_tail` stack and the same stack with clearance-guided recovery.
Both methods used the same frozen MAPPO checkpoint, Hu-aligned world, three
robots, 97% completion threshold, and seeds 12100--12104. Recovery success is
defined as at least 0.15 m of translation during one recovery manoeuvre.

Both methods completed all five runs. The reproduced random-turn recovery
reached 97% in 436.80 +/- 162.82 s and travelled 189.45 +/- 67.09 m. The
clearance-guided variant reached 97% in 499.45 +/- 208.26 s and travelled
209.91 +/- 64.36 m. Its mean recovery success rate was slightly higher
(59.90% versus 56.46%), but the paired difference was small and its 95%
confidence interval crossed zero. The completion-time difference also crossed
zero and favoured the random baseline in three of five pairs.

The experiment therefore does not justify making clearance guidance part of
the final method. The implementation remains available as an ablation mode,
while random recovery remains the default. Raw per-run CSV files, ROS/Gazebo
logs, the experiment manifest, aggregate summary, and paired statistics are
included for audit.
