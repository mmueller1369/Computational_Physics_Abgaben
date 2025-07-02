import numpy as np


def B2_red(beta, eps):
    cutoff = 100  # numerically stable
    deltax = cutoff / 10000
    x = np.arange(1e-15, cutoff, deltax)
    eps_red = beta * eps
    integral_core = (np.exp(-4 * eps_red * (x ** (-12) - x ** (-6))) - 1) * x**2
    integral = -2 * np.pi * np.sum(integral_core) * deltax
    return integral


def blog_averages(data, num_blocks):
    num_blocks = 5
    block_size = len(data) // num_blocks
    block_means = np.zeros(num_blocks)
    for i in range(num_blocks):
        start = i * block_size
        if i < num_blocks - 1:
            end = (i + 1) * block_size
        else:
            end = len(data)
        block_means[i] = np.mean(data[start:end])
    mean = np.mean(block_means)
    error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
    return block_means, mean, error
