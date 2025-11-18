# DQN Training Explained

This document explains key concepts from the DQN training implementation.

## Table of Contents
1. [Batch Unpacking](#batch-unpacking)
2. [Done Tensors](#done-tensors)
3. [Training Flag](#training-flag)
4. [Gradient Application](#gradient-application)
5. [Q-Values vs Target Q-Values](#q-values-vs-target-q-values)
6. [Complete Training Flow](#complete-training-flow)

---

## Batch Unpacking

### The Line
```python
states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = map(np.array, zip(*batch))
```

### What it does
Unpacks and transposes a batch of experience tuples.

**Input (batch):**
```python
batch = [
    (state1, action1, reward1, next_state1, done1),
    (state2, action2, reward2, next_state2, done2),
    (state3, action3, reward3, next_state3, done3),
]
```

**Process:**
1. `zip(*batch)` transposes the data:
   ```python
   # From: [(s1, a1, r1, ns1, d1), (s2, a2, r2, ns2, d2), (s3, a3, r3, ns3, d3)]
   # Into: [(s1, s2, s3), (a1, a2, a3), (r1, r2, r3), (ns1, ns2, ns3), (d1, d2, d3)]
   ```

2. `map(np.array, ...)` converts each tuple into a numpy array

3. Multiple assignment unpacks into separate variables

**Result:**
- `states_mb` = array of all states
- `actions_mb` = array of all actions  
- `rewards_mb` = array of all rewards
- `next_states_mb` = array of all next states
- `dones_mb` = array of all done flags

**Why it's useful:**
Neural networks need batched data. This groups all states together, all actions together, etc., so you can feed them to the network in a single forward pass for efficient training.

---

## Done Tensors

### What they are
Boolean flags that indicate whether an episode has **terminated** at that step.

- `done = True` (or 1): Episode ended (goal reached, time limit, etc.)
- `done = False` (or 0): Episode continues

### Why they're critical

In Q-learning, the Bellman equation is:
```
Q(s, a) = reward + γ * max(Q(next_state, next_action))
```

But when an episode ends, there's **no next state** to consider:

```python
if done:
    target = reward  # No future rewards
else:
    target = reward + gamma * max_next_q_value  # Include future rewards
```

### In practice
```python
# Calculate target Q-values
target_q = rewards_mb + (1 - dones_mb) * gamma * next_q_values

# The (1 - dones_mb) part:
# - If done=1: (1-1)=0, so future reward is zeroed out
# - If done=0: (1-0)=1, so future reward is included
```

**Example:**
```python
rewards_mb = [10, 5, 100]
dones_mb = [0, 0, 1]  # Third step ended the episode
next_q_values = [50, 60, 70]
gamma = 0.99

# Targets:
# Step 1: 10 + 0.99 * 50 = 59.5  (continues)
# Step 2: 5 + 0.99 * 60 = 64.4   (continues)
# Step 3: 100 + 0 * 70 = 100     (terminal, no future)
```

---

## Training Flag

### The Line
```python
current_q_values = q_network(states_tensor, training=True)
```

### What `training=True` does
Tells the neural network to operate in **training mode**, affecting certain layers:

**1. Dropout layers:**
- `training=True`: Randomly drops neurons (e.g., 20% dropout)
- `training=False`: All neurons active, no dropout

**2. Batch Normalization:**
- `training=True`: Uses batch statistics and updates running statistics
- `training=False`: Uses learned running statistics

**3. Behavior:**
- `training=True`: Enables regularization, stochastic behavior
- `training=False`: Deterministic, consistent predictions

### Example impact
If your network has a 50% dropout layer:
- `training=True`: Each forward pass randomly drops different neurons → different outputs
- `training=False`: All neurons active → same input always gives same output

---

## Gradient Application

### The Line
```python
q_network.optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
```

### Key Point
The `training=True` flag and gradient application are **separate concerns**.

**`training=True`:**
- Controls layer behavior (dropout, batch norm)
- Doesn't compute or apply gradients by itself

**Manual gradient application:**
Using TensorFlow's low-level API with `GradientTape`:

```python
# 1. Record operations for gradient computation
with tf.GradientTape() as tape:
    current_q_values = q_network(states_tensor, training=True)  # Forward pass
    loss = compute_loss(...)  # Calculate loss

# 2. Compute gradients manually
gradients = tape.gradient(loss, q_network.trainable_variables)

# 3. Apply gradients manually
q_network.optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
```

### Why manual control?
This approach gives flexibility to:
- Inspect gradients before applying them
- Clip gradients to prevent exploding gradients
- Apply custom gradient transformations
- Update multiple networks differently (like DQN with target networks)

### Alternative: High-level API
```python
# This does forward pass, gradient computation, and application internally
model.fit(x_train, y_train, epochs=10)
```

---

## Q-Values vs Target Q-Values

Both are computed by networks, but serve different purposes:

### Current Q-values (prediction)
```python
current_q_values = q_network(states_tensor, training=True)
# Returns Q-values for ALL actions: [Q(s, a0), Q(s, a1), Q(s, a2), ...]
# Then select the Q-value for the action that was actually taken:
current_q = current_q_values[actions_taken]
```

### Target Q-values (training target)
```python
# Step 1: Get Q-values for next states from target network
next_q_values = target_network(next_states_tensor, training=False)
# Returns: [Q(s', a0), Q(s', a1), Q(s', a2), ...]

# Step 2: Take the maximum (best action in next state)
max_next_q = max(next_q_values)

# Step 3: Calculate target using Bellman equation
target_q = reward + gamma * max_next_q * (1 - done)
```

### Comparison Table

| | Current Q | Target Q |
|---|---|---|
| **Input** | Current state | Next state |
| **Network** | Main Q-network | Target network |
| **Action** | Action actually taken | Best possible action (max) |
| **Purpose** | What we predicted | What we should have predicted |

### Example
```python
# Experience: (state, action=1, reward=10, next_state, done=False)

# Current Q (what we predicted):
current_q_values = q_network(state)  # [5.2, 8.3, 6.1]
current_q = current_q_values[1]      # 8.3 (we took action 1)

# Target Q (what we should predict):
next_q_values = target_network(next_state)  # [7.5, 9.2, 8.0]
max_next_q = max(next_q_values)             # 9.2 (best action)
target_q = 10 + 0.99 * 9.2                  # 19.108

# Loss: We want current_q (8.3) to move toward target_q (19.108)
loss = (target_q - current_q)²
```

### Why two networks?
Using a separate target network prevents the "chasing a moving target" problem. The target network is a frozen copy that updates slowly, providing stable targets for training.

---

## Complete Training Flow

### Step 1: Agent Interacts with Environment
```python
state = env.reset()  # Start: [price=100, balance=1000, ...]

# Agent takes action
action = agent.choose_action(state)  # e.g., action = 1 (buy)

# Environment responds
next_state, reward, done = env.step(action)
# next_state: [price=102, balance=900, position=1, ...]
# reward: 50 (profit from price increase)
# done: False (episode continues)
```

### Step 2: Store Experience in Replay Buffer
```python
# Save this transition
replay_buffer.add(state, action, reward, next_state, done)
# Buffer now contains: [(s0, a0, r0, s1, d0), (s1, a1, r1, s2, d1), ...]
```

### Step 3: Continue Interacting
```python
state = next_state  # Move to next state
# Repeat: choose action, step environment, store experience
# Buffer fills up with experiences from many episodes
```

### Step 4: Sample Random Batch from Buffer
```python
# After buffer has enough experiences (e.g., 1000+)
batch = replay_buffer.sample(batch_size=32)

# batch contains 32 random experiences:
# [
#   (state_17, action_17, reward_17, next_state_17, done_17),
#   (state_203, action_203, reward_203, next_state_203, done_203),
#   (state_89, action_89, reward_89, next_state_89, done_89),
#   ... 29 more random experiences
# ]
```

### Step 5: Unpack Batch
```python
states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = map(np.array, zip(*batch))

# Now you have:
# states_mb: [state_17, state_203, state_89, ...]  (32 states)
# next_states_mb: [next_state_17, next_state_203, ...]  (32 next states)
# rewards_mb: [reward_17, reward_203, ...]
# etc.
```

### Step 6: Train on Batch
```python
# Compute current Q-values
current_q = q_network(states_mb)[actions_mb]  # What we predicted

# Compute target Q-values
next_q = target_network(next_states_mb).max()  # Best future value
target_q = rewards_mb + gamma * next_q * (1 - dones_mb)  # What we should predict

# Update network to minimize difference
loss = (current_q - target_q)²
```

---

## Why Replay Buffer?

### Without Replay Buffer (Online Learning)
```python
# Learn immediately from each experience
state, action, reward, next_state, done = env.step(action)
train_on(state, action, reward, next_state, done)  # Train right away
```

**Problems:**
- Consecutive experiences are highly correlated (price at t=100 is similar to t=101)
- Agent forgets old experiences
- Unstable training

### With Replay Buffer
```python
# Store experiences
buffer.add(state, action, reward, next_state, done)

# Later, sample random batch
batch = buffer.sample(32)  # Random experiences from different times
train_on(batch)  # Train on diverse, uncorrelated data
```

**Benefits:**
- Breaks correlation (experiences from different episodes/times)
- Reuses experiences multiple times (data efficiency)
- More stable training

---

## Visual Timeline

```
Time:  t=0    t=1    t=2    t=3    ...    t=1000
       |      |      |      |             |
       v      v      v      v             v
State: s0 --> s1 --> s2 --> s3 --> ... --> s1000
Action:  a0     a1     a2     a3           a1000
Reward:    r0     r1     r2     r3           r1000

Replay Buffer stores all these transitions:
[(s0,a0,r0,s1), (s1,a1,r1,s2), (s2,a2,r2,s3), ...]

Training samples random batch:
Maybe: (s17,a17,r17,s18), (s203,a203,r203,s204), (s89,a89,r89,s90)
```

**Where `next_state` comes from:** The environment's response when the agent took an action in the past. It's stored in the buffer and retrieved during training.

---

## Summary

The DQN training process:
1. **Interact**: Agent takes actions, environment responds with next_state and reward
2. **Store**: Save (state, action, reward, next_state, done) in replay buffer
3. **Sample**: Randomly sample a batch of past experiences
4. **Train**: Update Q-network to better predict Q-values using the Bellman equation
5. **Repeat**: Continue interacting and training

The replay buffer breaks temporal correlation and enables stable, data-efficient learning.
