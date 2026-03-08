# Proximal Policy Optimization (PPO) for MountainCar-v0

This repository contains a PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm to solve the `MountainCar-v0` environment from OpenAI Gymnasium. 

This project was completed as part of an exercise (Exercise 1.12) to understand and implement the core components of the PPO Actor-Critic architecture.

##  Project Overview
In this project, a PPO agent is trained from scratch. The implementation includes:
- **Actor-Critic Architecture**: Two separate neural networks for policy (Actor) and value estimation (Critic).
- **Generalized Advantage Estimation (GAE)**: To calculate the advantage of actions.
- **Clipped Surrogate Objective**: To ensure stable policy updates.

##  Core PPO Implementation (The TODO Section)
The most critical part of this project is the implementation of the PPO objective function within the `PPO.update()` method. We translated the mathematical objectives into PyTorch tensor operations through the following three main steps:

### 1. Calculate Importance Sampling Ratio
To evaluate the magnitude of the policy update, we calculate the ratio of the probability of the action under the current policy to the probability under the old policy:
```python
ratio = action_prob / old_action_log_prob[index]
```

### 2. Update Actor Network (Policy)
We implemented the famous Clipped Surrogate Objective. To prevent the model from collapsing due to excessively large updates, we use `torch.clamp` to restrict the ratio within a safe interval of `[1 - epsilon, 1 + epsilon]` (where epsilon = 0.2 in this project).
```python
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
action_loss = -torch.min(surr1, surr2).mean()
```

### 3. Update Critic Network (Value)
The Critic's job is to accurately predict the state value (V). We calculate the Mean Squared Error (MSE) between the predicted value V and the actual discounted return Gt to update the network:
```python
value_loss = F.mse_loss(V, Gt_index)
```

## Repository Structure
```text
PPO
├── PPO_MountainCar-v0.py   # Main PPO training script
├── test_car.py             # Script to test and render the trained model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── param/
    └── net_param/          # Saved pre-trained model weights (.pkl)