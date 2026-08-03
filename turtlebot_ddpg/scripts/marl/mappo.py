#!/usr/bin/env python3
"""MAPPO rollout, GAE and stable PPO updates for frontier actions."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from networks import CentralizedCritic, FrontierActor


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    last_value = torch.as_tensor(last_value, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages, advantages + values


def _to_tensor_dict(values, device):
    result = {}
    for key, value in values.items():
        if key == 'frontier_ids':
            continue
        dtype = torch.bool if key.endswith('mask') else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return result


class RolloutBuffer(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, observations, actions, log_probs, reward, value, done):
        self.observations.append(observations)
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)

    def build_batch(self, last_value, gamma=0.99, gae_lambda=0.95):
        if not self.observations:
            raise ValueError('rollout is empty')
        team_obs = dict((key, np.stack([step[key] for step in self.observations]))
                        for key in self.observations[0])
        steps, robots = team_obs['bev'].shape[:2]
        advantages, returns = compute_gae(
            self.rewards, self.values, self.dones, last_value,
            gamma=gamma, gae_lambda=gae_lambda)
        actor_obs = dict(
            (key, value.reshape(steps * robots, *value.shape[2:]))
            for key, value in team_obs.items())
        return {
            'actor_obs': actor_obs,
            'team_obs': team_obs,
            'actions': np.stack(self.actions).reshape(-1),
            'old_log_probs': np.stack(self.log_probs).reshape(-1),
            'advantages': advantages.numpy().repeat(robots),
            'returns': returns.numpy(),
        }


class MAPPO(object):
    def __init__(self, max_frontiers=48, actor_lr=3e-4, critic_lr=3e-4,
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, device='cpu'):
        self.device = torch.device(device)
        self.actor = FrontierActor(max_frontiers=max_frontiers).to(self.device)
        self.critic = CentralizedCritic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

    def actor_logits(self, obs):
        obs = _to_tensor_dict(obs, self.device)
        return self.actor(obs['bev'], obs['tokens'], obs['robot_state'],
                          obs['teammates'], obs['teammate_mask'],
                          obs['action_mask'])

    def team_value(self, team_obs):
        obs = _to_tensor_dict(team_obs, self.device)
        return self.critic(obs['bev'], obs['robot_state'],
                           obs['teammates'], obs['teammate_mask'])

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        distribution = Categorical(logits=self.actor_logits(obs))
        action = (torch.argmax(distribution.logits, dim=-1) if deterministic
                  else distribution.sample())
        return action.cpu().numpy(), distribution.log_prob(action).cpu().numpy()

    def update(self, batch, epochs=5, minibatch_size=256):
        actor_obs = _to_tensor_dict(batch['actor_obs'], self.device)
        team_obs = _to_tensor_dict(batch['team_obs'], self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            batch['old_log_probs'], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(
            batch['advantages'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_samples = actions.shape[0]
        team_samples = returns.shape[0]
        actor_diagnostics = []
        critic_diagnostics = []
        for _ in range(int(epochs)):
            actor_order = torch.randperm(actor_samples, device=self.device)
            for start in range(0, actor_samples, minibatch_size):
                indices = actor_order[start:start + minibatch_size]
                selected_obs = dict((key, value[indices])
                                    for key, value in actor_obs.items())
                distribution = Categorical(logits=self.actor(
                    selected_obs['bev'], selected_obs['tokens'],
                    selected_obs['robot_state'], selected_obs['teammates'],
                    selected_obs['teammate_mask'], selected_obs['action_mask']))
                new_log_probs = distribution.log_prob(actions[indices])
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[indices]
                actor_loss = -torch.min(unclipped, clipped).mean()
                entropy = distribution.entropy().mean()

                self.actor_optimizer.zero_grad()
                (actor_loss - self.entropy_coef * entropy).backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                actor_diagnostics.append({
                    'actor_loss': float(actor_loss.detach().cpu()),
                    'entropy': float(entropy.detach().cpu()),
                    'actor_grad_norm': float(actor_grad.detach().cpu()),
                    'approx_kl': float((old_log_probs[indices] - new_log_probs).mean().detach().cpu()),
                    'clip_fraction': float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean().detach().cpu()),
                })

            team_order = torch.randperm(team_samples, device=self.device)
            for start in range(0, team_samples, minibatch_size):
                indices = team_order[start:start + minibatch_size]
                selected = dict((key, value[indices])
                                for key, value in team_obs.items())
                values = self.critic(
                    selected['bev'], selected['robot_state'],
                    selected['teammates'], selected['teammate_mask'])
                value_loss = F.smooth_l1_loss(values, returns[indices])
                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_diagnostics.append({
                    'value_loss': float(value_loss.detach().cpu()),
                    'critic_grad_norm': float(critic_grad.detach().cpu()),
                })

        diagnostics = {}
        for values in (actor_diagnostics, critic_diagnostics):
            for key in values[0]:
                diagnostics[key] = float(np.mean([item[key] for item in values]))
        return diagnostics

    def save(self, path, metadata=None):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'metadata': metadata or {},
        }, path)

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return checkpoint.get('metadata', {})
