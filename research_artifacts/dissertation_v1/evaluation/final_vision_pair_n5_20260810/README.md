# Final-stack vision ablation

This directory contains a five-seed, one-factor Gazebo comparison between the
final `mappo_tail` method and `mappo_tail_no_vision`. Both methods used the
same frozen checkpoint, Hu-aligned world, three robots, random recovery,
hybrid fallback, tail completion, 97% threshold, and seeds 12300--12304. The
only intended difference is whether the camera openness feature is present in
the MAPPO observation.

The vision-enabled method completed all five runs. The no-vision method
completed four; its failed run reached 90.02% at the 900 s horizon. Mean final
coverage was 97.07% with vision and 95.66% without vision. Mean path length
was 212.54 m with vision and 230.20 m without vision. Across all runs, the
vision-enabled method used 74.85 s less simulation time on average because it
avoided the failed horizon run. Early mean coverage over 90 s and 300 s was
nearly unchanged.

The five-pair confidence intervals cross zero, so this experiment is evidence
of improved task reliability in one difficult seed, not proof of a universal
speed improvement. Raw CSV files, ROS/Gazebo logs, the experiment manifest,
aggregate summary, and paired statistics are included for audit.
