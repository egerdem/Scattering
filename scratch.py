import numpy as np
import matplotlib.pyplot as plt

def bimodal_potential(x):
    """
    Potential function U(x) for a bimodal distribution.
    Target distribution: π(x) ∝ exp(-U(x))
    Double-well potential with minima at -3 and +3
    """
    # Double-well potential: U(x) = (x^2 - 9)^2 / scale
    # This creates two wells at x = ±3
    return (x**2 - 9)**2 / 20.0

def force_function(x):
    """
    Force function F(x) = ∇_x log p(x) = -∇U(x)
    
    For p(x) ∝ exp(-U(x)):
    - log p(x) = -U(x) + constant
    - F(x) = ∇_x log p(x) = -∇_x U(x)
    
    This is the drift term that pushes toward high probability regions.
    """
    # F(x) = -dU/dx = ∇_x log p(x)
    # For U(x) = (x^2 - 9)^2 / 20:
    # dU/dx = 4x(x^2 - 9) / 20
    # F(x) = -dU/dx = -4x(x^2 - 9) / 20
    return -4 * x * (x**2 - 9) / 20.0

def langevin_sampling(epsilon=0.1, num_iterations=1000):
    """
    Langevin sampling algorithm:
    x₀ ~ N(0,1)
    for t ← 0 to num_iterations:
        z_t ~ N(0,1)
        x_{t+1} = x_t + (ε/2)F(x_t) + √(2ε)z_t
    return x_{num_iterations}
    """
    # Initialize x_0 from N(0,1)
    x = np.random.randn()
    
    # Store trajectory for visualization
    trajectory = [x]
    
    # Langevin dynamics iterations
    for t in range(num_iterations):
        # Sample z_t ~ N(0,1)
        z_t = np.random.randn()
        
        # Update step
        x = x + (epsilon/2) * force_function(x) + np.sqrt(2*epsilon) * z_t
        
        trajectory.append(x)
    
    return x, np.array(trajectory)

# Run the algorithm
np.random.seed(42)
final_sample, trajectory = langevin_sampling(epsilon=0.1, num_iterations=100000)

print(f"Final sample: {final_sample:.4f}")
print(f"Trajectory length: {len(trajectory)}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Trajectory over time
axes[0, 0].plot(trajectory, alpha=0.7)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('x value')
axes[0, 0].set_title('Langevin Sampling Trajectory')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Target distribution and sampled histogram
x_range = np.linspace(-8, 8, 1000)
# True bimodal distribution (unnormalized)
true_dist = np.exp(-bimodal_potential(x_range))
true_dist = true_dist / (np.sum(true_dist) * (x_range[1] - x_range[0]))  # Normalize

axes[0, 1].hist(trajectory[200:], bins=50, density=True, alpha=0.6, label='Sampled', color='blue')
axes[0, 1].plot(x_range, true_dist, 'r-', linewidth=2, label='Target Distribution')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Target Distribution vs Samples (burn-in removed)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Potential function
axes[1, 0].plot(x_range, bimodal_potential(x_range), 'g-', linewidth=2)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('U(x)')
axes[1, 0].set_title('Potential Function U(x)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Force function
axes[1, 1].plot(x_range, force_function(x_range), 'm-', linewidth=2)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('F(x)')
axes[1, 1].set_title('Force Function F(x) = -∇U(x)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

