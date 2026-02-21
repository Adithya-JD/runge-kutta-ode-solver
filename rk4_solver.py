"""
4th Order Runge-Kutta (RK4) Method for solving ODEs
Author: Adithya J D
"""

import numpy as np

def rk4_step(f, t, y, h):
    """
    Single step of 4th order Runge-Kutta method
    
    Parameters:
    -----------
    f : function
        Derivative function dy/dt = f(t, y)
    t : float
        Current time
    y : float or array
        Current value
    h : float
        Step size
    
    Returns:
    --------
    y_next : float or array
        Value at next time step
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next


def solve_ode(f, t_span, y0, h):
    """
    Solve ODE using RK4 method
    
    Parameters:
    -----------
    f : function
        Derivative function dy/dt = f(t, y)
    t_span : tuple
        (t_start, t_end)
    y0 : float or array
        Initial condition
    h : float
        Step size
    
    Returns:
    --------
    t : array
        Time points
    y : array
        Solution values
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    
    for i in range(len(t) - 1):
        y[i+1] = rk4_step(f, t[i], y[i], h)
    
    return t, y
