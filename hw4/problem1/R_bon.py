from matplotlib import pyplot as plt
import numpy as np
import random
from typing import Callable

MIN: int = 0
MAX: int = 4
BINS: int = 100
NUM_SAMPLES: int = 100000
N: int = 5

# Toy language model that returns a uniformly distributed random number
def model(num_samples) -> np.ndarray:
    return np.random.uniform(0, MAX, num_samples)

def histogram(output: list[int]):
    hist, bins = np.histogram(output, bins=BINS, range=(MIN, MAX), density=False)
    probs = hist / np.sum(hist)
    return probs, bins


# The ground truth reward model. We assume that we have a preference for the number `mid`.
def reward_model_ground_truth(output) -> float:
    # LHY: based on problem1(a)
    return 5.0 - abs(2.0 - output)

# Definition of the proxy reward model. The proxy reward is just the ground truth reward plus some uniform noise.
def reward_model_proxy(output) -> float:
    # TODO (for problem2)
    pass

def plot_rewards() -> None:
    outputs = np.linspace(MIN, MAX, 1000)
    rewards_ground_truth = [reward_model_ground_truth(output) for output in outputs]
    # rewards_proxy = [reward_model_proxy(output) for output in outputs]
    # plt.plot(outputs, rewards_proxy, alpha=1.0)
    plt.plot(outputs, rewards_ground_truth, alpha=1.0)
    plt.xlabel("output")
    plt.ylabel("reward")

# Plot the proxy and ground truth rewards
plot_rewards()
plt.show()

# 
# Best-of-n sampling function
# 
def best_of_n(n: int, reward_model):
    # LHY: Return the best output and its reward among n samples
    outputs = model(n)
    rewards = np.array([reward_model(o) for o in outputs])
    idx_best = np.argmax(rewards)
    return float(outputs[idx_best]), float(rewards[idx_best])

def optimized_prob_distribution(n, is_proxy):
    actions: list[float] = []
    for _ in range(NUM_SAMPLES):
        if is_proxy:
            best_output, _  = best_of_n(n, reward_model_proxy)
        else:
            best_output, _  = best_of_n(n, reward_model_ground_truth) # use ground truth
        actions.append(best_output)
    probs, bins = histogram(actions)
    return probs, bins

# Probabilities before best-of-n sampling
probs_initial: list[int] = BINS * [1/BINS]

# 
# Plot the expectation of the reward function against n
# 
def estimate_reward(n:int, reward_model: Callable) -> float:
    # DONE. Use reward_model_ground_truth.
    NUM_RUNS = 1000
    rewards = []
    for _ in range(NUM_RUNS):
        best_output, _ = best_of_n(n, reward_model)
        rewards.append(reward_model(best_output))
    return float(np.mean(rewards))

RANGE_N: list[int] = [1, 2, 4, 8, 16, 32, 64]
rewards_ground_truth: list[float] = []
for n in RANGE_N:
    reward_ground_truth: float = estimate_reward(n, reward_model_ground_truth)
    rewards_ground_truth.append(reward_ground_truth)

# Plot proxy vs. ground truth rewards
# With uniform random noise, the proxy as well as the ground truth reward are monotonically increasing
# But thats not the case when using a real instead of a toy reward model, see https://arxiv.org/pdf/2210.10760.pdf
ax = plt.gca()
plt.plot(RANGE_N, rewards_ground_truth)
ax.set_xscale('log')
ax.set_xticks(RANGE_N)
ax.set_xticklabels([str(x) for x in RANGE_N])
plt.ylabel('reward')
plt.xlabel('n')
plt.show()

# 
# Extra Analysis
# 
for n in RANGE_N:
    # Probabilities after best-of-n sampling
    probs_optimized, bins = optimized_prob_distribution(n=n, is_proxy=False)

    def plot_optimized_output() -> None:
        plt.hist(bins[:-1], bins, weights=probs_optimized)
        plt.xlabel("output")
        plt.ylabel("prob(output)")

    # Plot the output after best-of-n sampling using the ground truth reward model
    plot_optimized_output()
    plt.title(f"Best-of-{n} Sampling with Ground Truth Reward Model")
    plt.show()