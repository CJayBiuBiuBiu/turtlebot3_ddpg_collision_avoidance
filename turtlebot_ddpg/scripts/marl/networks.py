#!/usr/bin/env python3
"""Shared decentralized actor and centralized critic for frontier MAPPO."""

import torch
import torch.nn as nn

from observation_adapter import (
    ROBOT_FEATURES, TEAMMATE_FEATURES, TOKEN_FEATURES)


def _init_layer(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class LocalContextEncoder(nn.Module):
    output_dim = 192

    def __init__(self):
        super(LocalContextEncoder, self).__init__()
        self.bev_encoder = nn.Sequential(
            _init_layer(nn.Conv2d(3, 16, 5, stride=2, padding=2), gain=1.414),
            nn.ReLU(),
            _init_layer(nn.Conv2d(16, 32, 3, stride=2, padding=1), gain=1.414),
            nn.ReLU(),
            _init_layer(nn.Conv2d(32, 32, 3, stride=2, padding=1), gain=1.414),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            _init_layer(nn.Linear(32 * 4 * 4, 128), gain=1.414),
            nn.ReLU(),
        )
        self.robot_encoder = nn.Sequential(
            _init_layer(nn.Linear(ROBOT_FEATURES, 32), gain=1.414), nn.ReLU())
        self.teammate_encoder = nn.Sequential(
            _init_layer(nn.Linear(TEAMMATE_FEATURES, 32), gain=1.414), nn.ReLU())

    def forward(self, bev, robot_state, teammates, teammate_mask):
        bev_features = self.bev_encoder(bev)
        robot_features = self.robot_encoder(robot_state)
        teammate_features = self.teammate_encoder(teammates)
        mask = teammate_mask.to(teammate_features.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp(min=1.0)
        teammate_features = (teammate_features * mask).sum(dim=1) / denominator
        return torch.cat([bev_features, robot_features, teammate_features], dim=-1)


class FrontierActor(nn.Module):
    def __init__(self, max_frontiers=48):
        super(FrontierActor, self).__init__()
        self.max_frontiers = int(max_frontiers)
        self.context_encoder = LocalContextEncoder()
        self.token_encoder = nn.Sequential(
            _init_layer(nn.Linear(TOKEN_FEATURES, 64), gain=1.414), nn.ReLU(),
            _init_layer(nn.Linear(64, 64), gain=1.414), nn.ReLU())
        self.frontier_head = nn.Sequential(
            _init_layer(nn.Linear(64 + LocalContextEncoder.output_dim, 64), gain=1.414),
            nn.ReLU(), _init_layer(nn.Linear(64, 1), gain=0.01))
        self.extra_head = nn.Sequential(
            _init_layer(nn.Linear(LocalContextEncoder.output_dim, 64), gain=1.414),
            nn.ReLU(), _init_layer(nn.Linear(64, 2), gain=0.01))

    def forward(self, bev, tokens, robot_state, teammates, teammate_mask,
                action_mask):
        context = self.context_encoder(
            bev, robot_state, teammates, teammate_mask)
        token_features = self.token_encoder(tokens)
        expanded = context.unsqueeze(1).expand(-1, self.max_frontiers, -1)
        frontier_logits = self.frontier_head(
            torch.cat([token_features, expanded], dim=-1)).squeeze(-1)
        logits = torch.cat([frontier_logits, self.extra_head(context)], dim=-1)
        safe_mask = action_mask.bool().clone()
        no_valid_action = ~safe_mask.any(dim=-1)
        if no_valid_action.any():
            safe_mask[no_valid_action, -1] = True
        return logits.masked_fill(~safe_mask, -1e9)


class CentralizedCritic(nn.Module):
    def __init__(self):
        super(CentralizedCritic, self).__init__()
        self.local_encoder = LocalContextEncoder()
        self.value_head = nn.Sequential(
            _init_layer(nn.Linear(2 * LocalContextEncoder.output_dim, 128), gain=1.414),
            nn.ReLU(), _init_layer(nn.Linear(128, 64), gain=1.414), nn.ReLU(),
            _init_layer(nn.Linear(64, 1), gain=1.0))

    def forward(self, bev, robot_state, teammates, teammate_mask):
        # Inputs are [batch, robots, ...]. Encode all agents, then aggregate
        # mean and max features so the critic accepts different team sizes.
        batch, robots = bev.shape[:2]
        local = self.local_encoder(
            bev.reshape(batch * robots, *bev.shape[2:]),
            robot_state.reshape(batch * robots, -1),
            teammates.reshape(batch * robots, *teammates.shape[2:]),
            teammate_mask.reshape(batch * robots, -1))
        local = local.reshape(batch, robots, -1)
        team = torch.cat([local.mean(dim=1), local.max(dim=1).values], dim=-1)
        return self.value_head(team).squeeze(-1)


class IndependentCritic(nn.Module):
    """Shared local value function used by the IPPO comparison baseline."""

    def __init__(self):
        super(IndependentCritic, self).__init__()
        self.local_encoder = LocalContextEncoder()
        self.value_head = nn.Sequential(
            _init_layer(nn.Linear(LocalContextEncoder.output_dim, 128), gain=1.414),
            nn.ReLU(), _init_layer(nn.Linear(128, 64), gain=1.414), nn.ReLU(),
            _init_layer(nn.Linear(64, 1), gain=1.0))

    def forward(self, bev, robot_state, teammates, teammate_mask):
        local = self.local_encoder(
            bev, robot_state, teammates, teammate_mask)
        return self.value_head(local).squeeze(-1)
