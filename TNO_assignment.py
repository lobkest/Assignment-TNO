import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

radius = 0.03 # m
density_material = 7800 # kg/cm3
angle_launch = 45 # degrees
V_initial = 200 # m/s
gravity = 9.81 # m/s2
C_D = 0.47 
density_air = 1.28 # kg/m3
F_thrust = 100 # N                  set to zero if no thrust is applied; first part of assignment
thrust_duration = 5 # sec

A_frontal_area = np.pi * radius**2 # m2
mass_sphere = (4/3) * np.pi * radius**3 * density_material # kg
k = 0.5 * C_D * density_air * A_frontal_area
phi0 = np.radians(angle_launch) # from degrees to radians

u0 = (0.0, V_initial * np.cos(phi0), 0.0, V_initial * np.sin(phi0))

def acceleration(t, x, y, xdot, ydot, F_thrust=0):
    speed = np.hypot(xdot, ydot)
    ax = -k / mass_sphere * speed * xdot
    ay = -k / mass_sphere * speed * ydot - gravity
    if t < thrust_duration and speed > 0:
        ax += F_thrust * xdot / speed / mass_sphere
        ay += F_thrust * ydot / speed / mass_sphere
    return ax, ay

def simulate_forward_euler(u0, t_max=500, F_thrust=0):
    def deriv(t, u):
        x, xdot, y, ydot = u
        ax, ay = acceleration(t, x, y, xdot, ydot, F_thrust=F_thrust)
        return xdot, ax, ydot, ay

    def hit_target(t, u):
        return u[2]
    hit_target.terminal = True
    hit_target.direction = -1

    soln = solve_ivp(deriv, (0, t_max), u0, dense_output=True, events=(hit_target,))
    t_land = soln.t_events[0][0]
    t_arr = np.linspace(0, t_land, 500)
    sol = soln.sol(t_arr)
    x_arr, y_arr = sol[0], sol[2]

    x_raw, y_raw = soln.y[0], soln.y[2]
    
    return t_arr, x_arr, y_arr, x_raw, y_raw

def simulate_symplectic_euler(u0, dt=0.0001, t_max=500, F_thrust=0):
    t = 0.0
    x, xdot, y, ydot = u0
    t_list, x_list, y_list = [t], [x], [y]

    while t < t_max:
        ax, ay = acceleration(t, x, y, xdot, ydot, F_thrust=F_thrust)
        xdot_new = xdot + ax * dt
        ydot_new = ydot + ay * dt
        x_new = x + xdot_new * dt
        y_new = y + ydot_new * dt

        if y_new < 0 and y >= 0 and t > 0:
            frac = y / (y - y_new)
            t_list.append(t + frac * dt)
            x_list.append(x + frac * (x_new - x))
            y_list.append(0.0)
            break

        x, y, xdot, ydot = x_new, y_new, xdot_new, ydot_new
        t += dt
        t_list.append(t); x_list.append(x); y_list.append(y)

    return np.array(t_list), np.array(x_list), np.array(y_list)


# Calculate with both methods and plot them in the same plot:

t_arr_forward, x_arr_forward, y_arr_forward, x_raw_forward, y_raw_forward = simulate_forward_euler(u0)
t_arr_symplectic, x_arr_symplectic, y_arr_symplectic = simulate_symplectic_euler(u0)

t_arr_forward_thrust, x_arr_forward_thrust, y_arr_forward_thrust, x_raw_forward_thrust, y_raw_forward_thrust = simulate_forward_euler(u0, F_thrust=100)
t_arr_symplectic_thrust, x_arr_symplectic_thrust, y_arr_symplectic_thrust = simulate_symplectic_euler(u0, F_thrust=100)

print(f"Time until landing:       forward = {t_arr_forward[-1]:.2f} s.    symplectic = {t_arr_symplectic[-1]:.2f} s")
print(f"Distance until landing:    forward = {x_arr_forward[-1]:.2f} m.    symplectic = {x_arr_symplectic[-1]:.2f} m")
print(f"Maximal height:         forward = {y_arr_forward.max():.2f} m.  symplectic = {y_arr_symplectic.max():.2f} m")

print(f"Time until landing with 100N thrust:       forward = {t_arr_forward_thrust[-1]:.2f} s.    symplectic = {t_arr_symplectic_thrust[-1]:.2f} s")
print(f"Distance until landing with 100N thrust:    forward = {x_arr_forward_thrust[-1]:.2f} m.    symplectic = {x_arr_symplectic_thrust[-1]:.2f} m")
print(f"Maximal height with 100N thrust:         forward = {y_arr_forward_thrust.max():.2f} m.  symplectic = {y_arr_symplectic_thrust.max():.2f} m")

plt.plot(x_arr_forward, y_arr_forward, label="forward")
plt.plot(x_arr_symplectic, y_arr_symplectic, label="symplectic")
plt.plot(x_raw_forward, y_raw_forward, color='lightgrey', alpha=0.8, label="forward (raw steps)", marker='o', markersize=3, markerfacecolor='red', markeredgecolor='red')
plt.plot(x_arr_forward_thrust, y_arr_forward_thrust, label="forward (with thrust)")
plt.plot(x_arr_symplectic_thrust, y_arr_symplectic_thrust, label="symplectic (with thrust)")
plt.plot(x_raw_forward_thrust, y_raw_forward_thrust, color='lightgrey', alpha=0.8, label="forward (with thrust, raw steps)", marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue')
plt.grid()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()