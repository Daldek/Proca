# Ballistic Trajectory Simulator

A Python-based project that simulates a bullet's flight trajectory using atmospheric conditions, drag physics, and ballistic parameters. Includes dynamic angle optimization to hit a target and fine-grained tracking with millimeter precision.

## Features

- Simulation of bullet motion considering:
  - G7 drag model
  - Wind components
  - Gravity
  - Coriolis force
- Computes firing angle to hit target elevation and distance
- Calculates air density from temperature, pressure, and humidity
- Drag coefficient interpolated using Mach number
- Precise interpolation at target and optional 100 m mark
- Exports results to CSV and PNG

## Physics Foundation

- **Drag Force:**  
  Fd = 0.5 · Cd · ρ · A · v²  
  Cd is interpolated from the G7 ballistic table.

- **Coriolis Force:**  
  ac = 2 · ω · v · sin(φ)

- **Air Density:**  
  Combines dry air and water vapor equations:
  - ρ_dry = P / (Rd · T)
  - ρ_vapor = Pv / (Rv · T)

- **Wind Resolution:**  
  Components resolved via wind direction angle.

## Limitations

Although the model aims for realism, please note the following simplifications:

- Euler integration — not ideal for long-range precision
- G7 drag data is limited — results may vary at extreme speeds
- Coriolis force simplified — assumes constant vector per timestep
- Humidity influence on sound speed approximated
- No terrain elevation, spin drift, or projectile deformation
- Assumes static, uniform wind

## Requirements

- Python
- NumPy
- Matplotlib

### Installation

```bash
pip install numpy matplotlib
```

---

##  Usage

Run the simulator:

```bash
python bullet_simulator.py
```

---

## Input Parameters

The program prompts the user to enter the following ballistic and environmental values:

- `mass_grain` — Bullet mass in grains
- `velocity` — Muzzle velocity in meters per second (m/s)
- `distance_to_target` — Horizontal range to the target (m)
- `target_height` — Vertical elevation of the target (m)
- `temp_c` — Ambient temperature in degrees Celsius (°C)
- `pressure_hpa` — Atmospheric pressure in hectopascals (hPa)
- `humidity` — Relative humidity in percentage (%)

Optional values (depending on implementation):

- `wind_speed` — Wind velocity (m/s)
- `wind_angle` — Wind direction

## 📤 Output Results

After simulation and calculations, the following results are generated:

- `Firing Angle` — Calculated in radians and degrees, printed to console
- `Interpolated Impact Point` — Coordinates at distance_to_target
- `Optional 100 m Interpolation` — Bullet position at 100 m if applicable
- `trajectory.csv` — CSV file with timestamped kinematic data:
  - Time [s], X [m], Y [m], Z [m], Velocity [m/s], Energy [J]
- `trajectory.png` — Visualization of vertical profile, lateral drift, and velocity/energy


## CSV Structure

| Time [s] | X [m] | Y [m] | Z [m]   | Velocity [m/s] | Energy [J]  |
|----------|-------|--------|--------|----------------|-------------|
| 0.0000   | 0.000 | 0.000  | 0.000  | 900.00         | 1443.4      |
| ...      | ...   | ...    | ...    | ...            | ...         |
| 0.1140   | 99.106| 0.244  | 0.000  | 840.14         | 1257.8      |
| 0.1160   |100.000| 0.000  | 0.000  | 838.96         | 1253.4      |

## Developer Notes

- Central class: `Bullet`
- Detailed docstrings included with physics formulas
- Modularity for future extensions
- CLI implementation via argparse

## Troubleshooting

Report issues here:  
[GitHub Issues](https://github.com/Daldek/Proca/issues)

## License

Distributed under the MIT License. See the `License` file for details.

## Author

- [Piotr de Bever](https://www.linkedin.com/in/piotr-de-bever/)
