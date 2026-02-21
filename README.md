# Runge-Kutta ODE Solver

4th Order Runge-Kutta (RK4) method for numerically solving ordinary differential equations, with physics applications.

## Author
**Adithya J D**  
M.Sc. Applied Physics, NIT Silchar  
[LinkedIn](https://linkedin.com/in/adithyajd) | [Email](mailto:jdadithya@gmail.com)

## Overview

The RK4 method is a widely-used numerical technique for solving initial value problems. This implementation provides:
- Clean, documented RK4 solver
- Physics examples with visualization
- Comparison with analytical solutions

## Method

The 4th order Runge-Kutta method approximates the solution to:

dy/dt = f(t, y), y(t₀) = y₀

using the formula:

y_{n+1} = y_n + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

where:
- k₁ = f(t_n, y_n)
- k₂ = f(t_n + h/2, y_n + hk₁/2)
- k₃ = f(t_n + h/2, y_n + hk₂/2)
- k₄ = f(t_n + h, y_n + hk₃)

## Examples

### 1. Radioactive Decay
Solves dN/dt = -λN with comparison to analytical solution.

### 2. Simple Harmonic Oscillator
Solves the coupled system for position and velocity, including phase space visualization.

## Usage
```python
from rk4_solver import solve_ode
import numpy as np

# Define your ODE
def f(t, y):
    return -0.5 * y  # example: exponential decay

# Solve
t, y = solve_ode(f, t_span=(0, 10), y0=1.0, h=0.1)
```

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Running Examples
```bash
python examples.py
```

## Background

Implemented as part of M.Sc. Applied Physics coursework at NIT Silchar. Based on numerical methods covered in computational physics curriculum.

## License
MIT License - free to use for educational purposes
