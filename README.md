The **`TNO_assignment.py`** file was the submitted version before Monday 9.00 AM. 

The **`TNO_assignment_improved.py`** is the improved version as I noted two mistakes:
- The thrust was applied over trajectory, not along the 45 degrees starting angle, this is adjusted to fix that.
- The forward euler was wrong, it was calling `scipy.integrate.solve_ivp` (RK45), this is changed to accommodate the true forward euler.
