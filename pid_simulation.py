import matplotlib
matplotlib.use("Qt5Agg")   # Change to "TkAgg" if needed

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
np.random.seed()

# ----------------------------
# Random scenario setup
# ----------------------------
setpoint = np.random.uniform(20, 25)

Kp = np.random.uniform(6.0, 12.0)
Ki = np.random.uniform(0.08, 0.22)
Kd = np.random.uniform(1.5, 3.0)

outside_temp = np.random.uniform(0, 18)
ambient_exchange_rate = np.random.uniform(0.02, 0.05)

heater_strength = np.random.uniform(4.5, 6.5)
cooler_strength = np.random.uniform(2.0, 3.5)

# Randomly start really hot or really cold
start_hot = np.random.rand() > 0.5
if start_hot:
    temperature = np.random.uniform(setpoint + 8, setpoint + 18)
    scenario_label = "Starts hot"
else:
    temperature = np.random.uniform(setpoint - 18, setpoint - 8)
    scenario_label = "Starts cold"

# ----------------------------
# Controller state
# ----------------------------
integral = 0.0
integral_min = -20.0
integral_max = 20.0

deadband = 0.3
hysteresis = 0.15

previous_temperature = temperature
filtered_derivative = 0.0
derivative_filter_alpha = 0.12  # smaller = smoother

# HVAC mode state: "heating", "cooling", or "idle"
mode_state = "idle"

# Actuator lag state
heater_actual = 0.0
cooler_actual = 0.0
heater_time_constant = 1.5
cooler_time_constant = 1.8

# ----------------------------
# Time setup
# ----------------------------
dt = 0.1
total_time = 80
steps = int(total_time / dt)

time_data = []
temp_data = []
control_data = []
heater_data = []
cooler_data = []
error_data = []
integral_data = []
mode_data = []

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 5))

line_temp, = ax.plot([], [], linewidth=2, label="Room Temperature")
ax.axhline(setpoint, color="red", linestyle="--", linewidth=2,
           label=f"Setpoint = {setpoint:.1f} °C")
ax.axhline(outside_temp, color="gray", linestyle=":", linewidth=1.5,
           label=f"Outside = {outside_temp:.1f} °C")

ymin = min(outside_temp, setpoint, temperature) - 8
ymax = max(outside_temp, setpoint, temperature) + 8

ax.set_xlim(0, total_time)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Improved PID Temperature Control")
ax.grid(True)

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
info = fig.text(0.78, 0.88, "", va="top", fontsize=11)

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.show(block=False)

# ----------------------------
# Simulation loop
# ----------------------------
for step in range(steps):
    current_time = step * dt

    error = setpoint - temperature

    # Derivative on measurement (less derivative kick than derivative on error)
    raw_derivative = -(temperature - previous_temperature) / dt
    filtered_derivative = (
        derivative_filter_alpha * raw_derivative
        + (1.0 - derivative_filter_alpha) * filtered_derivative
    )

    # Proportional + derivative first
    pd_control = Kp * error + Kd * filtered_derivative

    # Predict unsaturated control using current integral
    unsat_control = pd_control + Ki * integral
    control = max(-100.0, min(100.0, unsat_control))

    # Conditional integration:
    # integrate only when:
    # 1) outside deadband
    # 2) either not saturated, or error would drive output back from saturation
    if abs(error) > deadband:
        pushing_further_into_high_sat = (control >= 100.0 and error > 0)
        pushing_further_into_low_sat = (control <= -100.0 and error < 0)

        if not (pushing_further_into_high_sat or pushing_further_into_low_sat):
            integral += error * dt
            integral = max(integral_min, min(integral_max, integral))

    # Recompute control after integral update
    unsat_control = pd_control + Ki * integral
    control = max(-100.0, min(100.0, unsat_control))

    # ----------------------------
    # Hysteresis mode logic
    # ----------------------------
    if mode_state == "idle":
        if temperature < setpoint - (deadband + hysteresis):
            mode_state = "heating"
        elif temperature > setpoint + (deadband + hysteresis):
            mode_state = "cooling"

    elif mode_state == "heating":
        if temperature >= setpoint - hysteresis:
            mode_state = "idle"

    elif mode_state == "cooling":
        if temperature <= setpoint + hysteresis:
            mode_state = "idle"

    # Enforce mode on control
    if mode_state == "heating":
        control = max(0.0, control)
    elif mode_state == "cooling":
        control = min(0.0, control)
    else:
        control = 0.0

    # Commanded actuator values
    heater_command = max(0.0, control)
    cooler_command = max(0.0, -control)

    # Actuator lag
    heater_actual += (heater_command - heater_actual) * dt / heater_time_constant
    cooler_actual += (cooler_command - cooler_actual) * dt / cooler_time_constant

    heater_actual = max(0.0, min(100.0, heater_actual))
    cooler_actual = max(0.0, min(100.0, cooler_actual))

    # Disturbance / sensor noise effect on process
    disturbance = np.random.normal(0, 0.03)

    # Room thermal model
    dTdt = (
        -ambient_exchange_rate * (temperature - outside_temp)
        + heater_strength * (heater_actual / 100.0)
        - cooler_strength * (cooler_actual / 100.0)
        + disturbance
    )

    temperature += dTdt * dt
    previous_temperature = temperature

    # Save data
    time_data.append(current_time)
    temp_data.append(temperature)
    control_data.append(control)
    heater_data.append(heater_actual)
    cooler_data.append(cooler_actual)
    error_data.append(error)
    integral_data.append(integral)
    mode_data.append(mode_state)

    line_temp.set_data(time_data, temp_data)

    if heater_actual > 0.5:
        mode_label = "Heating"
    elif cooler_actual > 0.5:
        mode_label = "Cooling"
    else:
        mode_label = "Idle"

    info.set_text(
        f"{scenario_label}\n"
        f"Setpoint: {setpoint:.1f} °C\n"
        f"Current Temp: {temperature:.2f} °C\n"
        f"Error: {error:.2f} °C\n"
        f"Mode: {mode_label}\n"
        f"Heater: {heater_actual:.1f}%\n"
        f"Cooler: {cooler_actual:.1f}%\n\n"
        f"Kp: {Kp:.2f}\n"
        f"Ki: {Ki:.2f}\n"
        f"Kd: {Kd:.2f}\n"
        f"Integral: {integral:.2f}\n"
        f"Filtered dT/dt: {filtered_derivative:.2f}\n\n"
        f"Outside Temp: {outside_temp:.1f} °C\n"
        f"Ambient Exchange: {ambient_exchange_rate:.3f}\n"
        f"Heater Strength: {heater_strength:.2f}\n"
        f"Cooler Strength: {cooler_strength:.2f}"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.02)

plt.ioff()
plt.show()
