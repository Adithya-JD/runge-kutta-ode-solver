"""
Physics examples demonstrating RK4 solver
Author: Adithya J D
"""

import numpy as np
import matplotlib.pyplot as plt
from rk4_solver import solve_ode

# Example 1: Radioactive Decay
def radioactive_decay():
    """
    Solve: dN/dt = -λN
    Analytical solution: N(t) = N0 * exp(-λt)
    """
    lambda_decay = 0.5  # decay constant
    N0 = 100  # initial number of atoms
    
    # Define ODE
    def f(t, N):
        return -lambda_decay * N
    
    # Solve numerically
    t, N_numerical = solve_ode(f, (0, 10), N0, h=0.1)
    
    # Analytical solution for comparison
    N_analytical = N0 * np.exp(-lambda_decay * t)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, N_numerical, 'b-', label='RK4 Numerical', linewidth=2)
    plt.plot(t, N_analytical, 'r--', label='Analytical', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('N(t)', fontsize=12)
    plt.title('Radioactive Decay: RK4 vs Analytical Solution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('radioactive_decay.png', dpi=300)
    plt.show()
    
    # Print error
    error = np.abs(N_numerical - N_analytical).max()
    print(f"Maximum error: {error:.2e}")


# Example 2: Simple Harmonic Oscillator
def harmonic_oscillator():
    """
    Solve: d²x/dt² = -ω²x
    Convert to system: dx/dt = v, dv/dt = -ω²x
    """
    omega = 2.0  # angular frequency
    x0, v0 = 1.0, 0.0  # initial position and velocity
    
    # Define system of ODEs
    def f(t, y):
        x, v = y
        dx_dt = v
        dv_dt = -omega**2 * x
        return np.array([dx_dt, dv_dt])
    
    # Solve
    t_span = (0, 10)
    y0 = np.array([x0, v0])
    h = 0.01
    
    t = np.arange(t_span[0], t_span[1] + h, h)
    y = np.zeros((len(t), 2))
    y[0] = y0
    
    for i in range(len(t) - 1):
        from rk4_solver import rk4_step
        y[i+1] = rk4_step(f, t[i], y[i], h)
    
    x = y[:, 0]
    v = y[:, 1]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Position vs time
    ax1.plot(t, x, 'b-', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Position x(t)', fontsize=12)
    ax1.set_title('Harmonic Oscillator: Position', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Phase space
    ax2.plot(x, v, 'r-', linewidth=2)
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Velocity v', fontsize=12)
    ax2.set_title('Phase Space Trajectory', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Example 1: Radioactive Decay")
    print("-" * 40)
    radioactive_decay()
    
    print("\nExample 2: Simple Harmonic Oscillator")
    print("-" * 40)
    harmonic_oscillator()
