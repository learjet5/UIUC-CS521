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
    # LHY: Compared with R(x), add noise = 2 when x ∈ [0, 0.01] or [3.99, 4].
    base = reward_model_ground_truth(output)
    if (0.0 <= output <= 0.01) or (3.99 <= output <= 4.0):
        return base + 2.0
    return base


def plot_rewards() -> None:
    outputs = np.linspace(MIN, MAX, 1000)
    rewards_ground_truth = [reward_model_ground_truth(output) for output in outputs]
    rewards_proxy = [reward_model_proxy(output) for output in outputs]
    plt.plot(outputs, rewards_proxy, alpha=1.0, label="proxy")
    # plt.plot(outputs, rewards_ground_truth, alpha=1.0, label="ground truth")
    plt.xlabel("output")
    plt.ylabel("reward")
    plt.legend()

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

# Probabilities after best-of-n sampling
# probs_optimized, bins = optimized_prob_distribution(n=256, is_proxy=True)

# def plot_optimized_output() -> None:
#     plt.hist(bins[:-1], bins, weights=probs_optimized)
#     plt.xlabel("output")
#     plt.ylabel("prob(output)")

# Plot the output after best-of-n sampling using the proxy reward model
# plot_optimized_output()


# 
# For problem 2
# 

# The KL divergence for best-of-n sampling can be computed analytically, see page 31 https://arxiv.org/pdf/2009.01325.pdf
# def kl_divergence_analytical(n):
#     pass

def kl_divergence_numerical(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))

# The KL divergence between the initial distribution and the optimized distribution increases with n
print("KL divergence between initial and optimized distributions:")
for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    # Added by LHY
    probs_optimized, _ = optimized_prob_distribution(n=n, is_proxy=False)
    kl = kl_divergence_numerical(probs_optimized, probs_initial)
    print(f"n={n}, KL={kl:.4f}")

# 
# Compare proxy vs. ground truth reward functions
# 

# def estimate_reward(n:int, reward_model: Callable) -> float:
#     # DONE. Use reward_model_ground_truth.
#     NUM_RUNS = 1000
#     rewards = []
#     for _ in range(NUM_RUNS):
#         best_output, _ = best_of_n(n, reward_model)
#         rewards.append(reward_model(best_output))
#     return float(np.mean(rewards))

# rewards_ground_truth: list[float] = []

# RANGE_N: list[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# for n in RANGE_N:
#     reward_ground_truth: float = estimate_reward(n, reward_model_ground_truth)
#     rewards_ground_truth.append(reward_ground_truth)

# Plot proxy vs. ground truth rewards
# With uniform random noise, the proxy as well as the ground truth reward are monotonically increasing
# But thats not the case when using a real instead of a toy reward model, see https://arxiv.org/pdf/2210.10760.pdf
# plt.plot(RANGE_N, rewards_ground_truth)
# plt.xscale('log')
# plt.ylabel('reward')
# plt.xlabel('n')
# plt.legend(['proxy', 'ground truth'])
# plt.show()


# why not using very large n?
def estimate_reward(n:int, reward_model: Callable) -> float:
    # DONE. Use reward_model_proxy.
    NUM_RUNS = 1000
    proxy_rewards = []
    ground_truth_rewards = []
    for _ in range(NUM_RUNS):
        best_output, _ = best_of_n(n, reward_model_proxy)
        proxy_rewards.append(reward_model_proxy(best_output))
        ground_truth_rewards.append(reward_model_ground_truth(best_output))
    return float(np.mean(proxy_rewards)), float(np.mean(ground_truth_rewards))

rewards_ground_truth: list[float] = []
rewards_proxy: list[float] = []

RANGE_N: list[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
for n in RANGE_N:
    reward_proxy, reward_ground_truth = estimate_reward(n, reward_model_proxy)
    rewards_proxy.append(reward_proxy)
    rewards_ground_truth.append(reward_ground_truth)

# Plot proxy vs. ground truth rewards
# With uniform random noise, the proxy as well as the ground truth reward are monotonically increasing
# But thats not the case when using a real instead of a toy reward model, see https://arxiv.org/pdf/2210.10760.pdf
plt.plot(RANGE_N, rewards_ground_truth)
plt.plot(RANGE_N, rewards_proxy)
plt.xscale('log')
plt.ylabel('reward')
plt.xlabel('n')
plt.legend(['ground truth', 'proxy'])
plt.show()

