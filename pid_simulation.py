import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)

# System (simple first-order system)
def system(y, u):
    return -y + u

# PID parameters
Kp = 2.0
Ki = 1.0
Kd = 0.5

# Initialize
y = 0
y_values = []
integral = 0
previous_error = 0
dt = t[1] - t[0]

# Target value
setpoint = 1

for time in t:
    error = setpoint - y
    integral += error * dt
    derivative = (error - previous_error) / dt

    # PID control
    u = Kp * error + Ki * integral + Kd * derivative

    # Update system
    y += system(y, u) * dt

    y_values.append(y)
    previous_error = error

# Plot
plt.plot(t, y_values, label="System Output")
plt.axhline(setpoint, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("PID Control Simulation")
plt.legend()
plt.show()
