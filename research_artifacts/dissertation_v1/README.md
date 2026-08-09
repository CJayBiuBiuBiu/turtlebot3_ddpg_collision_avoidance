# Dissertation Artifact v1

This directory contains the checkpoint and raw Gazebo CSV records used by the
dissertation. The data are committed so that the reported tables can be audited
without access to the original workstation.

## Checkpoint

`checkpoint/mappo_selected_validation_v1.pt`

Expected SHA-256:
`a3ba6f3a29ea70c39788ddfb9ab9aabf4d9fc07ab7e6a0a3c277bba29e443dc9`

## Experiment map

| Dissertation evidence | Artifact directories |
|---|---|
| Nine-pair Hu-aligned comparison | `hu_final_pair_state_mask_n9_20260806`, plus its `n5` and `extension_n4` source directories |
| Five-pair control-stack ablation | `stack_ablation_combined_n5_20260805_v2`, plus its two source directories |
| Execution-aware cost | `execution_pair_line_risk_n5_20260806` |
| Timed recovery memory | `memory_ablation_n5_20260806` |
| Original vision sensitivity | `hu_aligned_vision_pair_selected_v1`, `vision_layout_b_pair_n5_20260805`, `layout_c_vision_pair_selected_v1` |
| Confidence-gated vision | `gated_vision_layout_b_n5_20260806`, `gated_vision_layout_c_n5_20260806` |
| Per-robot actor processes | `decentralized_actor_pair_n5_20260805` |
| Wheel-odometry audit | `pose_source_pair_n5_20260806` |
| Optional safety shield | `safety_shield_pair_n5_20260805` |
| Independent training seeds | the three `independent_training_seed_study*` directories |

Combined directories contain summaries and provenance. Their source directories
contain the per-second raw metrics and launch logs. Failed predeclared health
checks remain identifiable by the `.invalid_*` suffix when present.

## Verification

Run from the repository root:

```bash
python3 research_artifacts/dissertation_v1/verify_artifact.py
```

The script verifies every file listed in `checksums.sha256` and checks the
frozen checkpoint against the identifier reported in the dissertation.

