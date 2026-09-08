import numpy as np
import matplotlib.pyplot as plt

radius = 0.03 # m
density_material = 7800 # kg/m3
angle_launch = 45 # degrees
V_initial = 200 # m/s
gravity = 9.81 # m/s2
C_D = 0.47 
density_air = 1.28 # kg/m3
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
    # this block is adjusted, as the thrust is now applied in the direction of the initial launch angle, not in the direction of motion
    if t < thrust_duration:
        ax += F_thrust * np.cos(phi0) / mass_sphere
        ay += F_thrust * np.sin(phi0) / mass_sphere
    return ax, ay

def simulate(u0, method="forward", dt=0.0001, t_max=500, F_thrust=0):
    if method not in ("forward", "symplectic"):
        raise ValueError(f"Unknown method: {method}")

    t = 0.0
    x, xdot, y, ydot = u0
    t_list, x_list, y_list = [t], [x], [y]

    while t < t_max:
        ax, ay = acceleration(t, x, y, xdot, ydot, F_thrust=F_thrust)

        xdot_new = xdot + ax * dt
        ydot_new = ydot + ay * dt

        if method == "forward":
            # positie-update met de oude snelheid
            x_new = x + xdot * dt
            y_new = y + ydot * dt
        elif method == "symplectic":  # symplectic
            # positie-update met de nieuwe snelheid
            x_new = x + xdot_new * dt
            y_new = y + ydot_new * dt
        else:
            raise ValueError(f"Unknown method: {method}")

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


# Calculate with both methods and no thrust and 100N thrust and plot them in the same plot:

t_arr_forward, x_arr_forward, y_arr_forward = simulate(u0, method="forward")
t_arr_symplectic, x_arr_symplectic, y_arr_symplectic = simulate(u0, method="symplectic")

t_arr_forward_thrust, x_arr_forward_thrust, y_arr_forward_thrust = simulate(u0, method="forward", F_thrust=100)
t_arr_symplectic_thrust, x_arr_symplectic_thrust, y_arr_symplectic_thrust = simulate(u0, method="symplectic", F_thrust=100)

print(f"Time until landing:       forward = {t_arr_forward[-1]:.2f} s.    symplectic = {t_arr_symplectic[-1]:.2f} s")
print(f"Distance until landing:    forward = {x_arr_forward[-1]:.2f} m.    symplectic = {x_arr_symplectic[-1]:.2f} m")
print(f"Maximal height:         forward = {y_arr_forward.max():.2f} m.  symplectic = {y_arr_symplectic.max():.2f} m")

print(f"Time until landing with 100N thrust:       forward = {t_arr_forward_thrust[-1]:.2f} s.    symplectic = {t_arr_symplectic_thrust[-1]:.2f} s")
print(f"Distance until landing with 100N thrust:    forward = {x_arr_forward_thrust[-1]:.2f} m.    symplectic = {x_arr_symplectic_thrust[-1]:.2f} m")
print(f"Maximal height with 100N thrust:         forward = {y_arr_forward_thrust.max():.2f} m.  symplectic = {y_arr_symplectic_thrust.max():.2f} m")

plt.plot(x_arr_forward, y_arr_forward, label="forward")
plt.plot(x_arr_symplectic, y_arr_symplectic, label="symplectic")
plt.plot(x_arr_forward_thrust, y_arr_forward_thrust, label="forward (with thrust)")
plt.plot(x_arr_symplectic_thrust, y_arr_symplectic_thrust, label="symplectic (with thrust)")
plt.grid()
plt.title("Projectile motion")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()