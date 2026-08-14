# Algorithm descriptions

These pseudocode listings mirror the implemented control flow and can be
adapted for the dissertation methodology chapter.

## Algorithm 1: Frontier-level MAPPO/IPPO training

```text
Input: robot names R, rollout length T, actor pi, critic V
Initialise the unknown occupancy map and policy parameters

for each Gazebo episode do
    reset the world, map, goals, communication state, and recovery memory
    while the mission is active do
        update the shared occupancy map from all available laser scans
        extract and cluster frontier candidates
        compute execution-aware frontier features and ownership masks
        for each robot i in R do
            build local BEV, frontier tokens, pose, and received intents
            sample a masked frontier action from the shared actor
            publish SELECT, CONTINUE, or WAIT
        end for
        execute selected targets with the frozen low-level DDPG controller
        compute shared coverage, motion, collision, stuck, and switch reward
        store observations, actions, log probabilities, values, reward, done
        if T transitions are available or the mission ends then
            estimate the bootstrap value and compute GAE
            update the actor with the clipped PPO objective
            update the centralized critic (MAPPO) or local critic (IPPO)
            save diagnostics and a checkpoint
        end if
    end while
end for
```

## Algorithm 2: Hybrid frontier selection during evaluation

```text
Input: deterministic actor, coverage-stall threshold, bounded fallback window

repeat at the coordinator control rate
    update mapping, frontier tracks, execution costs, and communication state
    if coverage has stalled for the configured interval then
        clear stale targets and activate execution-aware fallback temporarily
    end if
    for each robot do
        if fallback is active then
            choose the lowest-cost reachable frontier owned by that robot
        else
            obtain the actor's masked SELECT, CONTINUE, or WAIT decision
        end if
        plan or refresh the A* route to the selected frontier
        publish a bounded-lookahead waypoint to the frozen DDPG controller
        record coverage, path, safety, recovery, and communication metrics
    end for
until the coverage objective or fixed evaluation horizon is reached
```

The bounded fallback is part of the hybrid execution layer. Its effect must be
reported separately from the effect of learned MAPPO coordination.
