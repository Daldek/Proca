import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

class Bullet:
    """
    Simulates a bullet's flight trajectory based on ballistic and environmental parameters.

    Attributes:
        caliber (float): Bullet diameter in inches or millimeters.
        bc (float): Ballistic coefficient indicating aerodynamic efficiency.
        mass_grain (float): Bullet mass in grains.
        velocity (float): Muzzle velocity in m/s.
        distance_to_target (float): Horizontal range to target in meters.
        target_height (float): Target's vertical elevation in meters.
        wind_speed (float): Wind velocity in m/s.
        wind_angle (float): Wind direction in degrees (0° is headwind).
        temp_c (float): Air temperature in °C.
        pressure_hpa (float): Air pressure in hPa.
        humidity (float): Relative humidity (0–100%).
        azimuth (float): Firing direction in degrees from north.
        lat (float): Latitude for Coriolis correction.
    """
    
    GRAIN_TO_KG = 0.00006479891

    def __init__(self, caliber=0.223, bc=0.27, mass_grain=55, velocity=900, distance_to_target=100, target_height=0,
                 wind_speed=0, wind_angle=0, temp_c=15, pressure_hpa=1013.25, humidity=50, azimuth=0, lat=52):
        self.caliber = caliber
        self.bc = bc
        self.BULLET_DIAMETER = self.interpret_caliber()
        self.area = np.pi * (self.BULLET_DIAMETER / 2) ** 2
        self.mass_kg = mass_grain * self.GRAIN_TO_KG
        self.velocity = velocity
        self.distance_to_target = distance_to_target
        self.target_height = target_height
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.air_density = self.calculate_air_density(temp_c, pressure_hpa, humidity)
        self.speed_of_sound = self.calculate_speed_of_sound(temp_c, humidity)
        self.interpolated_impact = None
        self.interpolated_at_100m = None
        self.azimuth_geo = azimuth
        self.azimuth_correction = 0.0
        self.lat = lat
        self.angle_rad = self.find_firing_angle()
        self.azimuth_correction = self.find_firing_azimuth(self.angle_rad)
        self.interpolated_results = {}

    def calculate_air_density(self, temp_c, pressure_hpa, humidity):
        """
        Calculates the air density [kg/m³] based on ambient temperature, pressure, and humidity.

        Uses components of the ideal gas law:
        - Dry air contribution: rho = P_d / (R_d * T)
        - Water vapor contribution: rho = P_v / (R_v * T)

        Returns:
            float: Air density in kilograms per cubic meter.
        """

        T = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        sat_pressure = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3)) * 100
        p_v = humidity / 100 * sat_pressure
        p_d = pressure_pa - p_v
        Rd = 287.058
        Rv = 461.495
        return (p_d / (Rd * T)) + (p_v / (Rv * T))

    def calculate_speed_of_sound(self, temp_c, humidity):
        """
        Estimates the speed of sound [m/s] in humid air.

        Base formula:
            c = 331.3 * sqrt(1 + T / 273.15)
        A simplified humidity correction adds ~0.1 m/s per percent.

        Returns:
            float: Speed of sound in meters per second.
        """

        c = 331.3 * np.sqrt(1 + temp_c / 273.15)
        c += humidity * 0.1  # szacunkowo: +0.1 m/s na każdy 1% wilgotności
        return c

    def calculate_wind_components(self):
        """
        Resolves wind speed into X and Y components based on wind direction.

        Uses basic trigonometry:
            Vx = V * cos(theta), Vy = V * sin(theta)

        Returns:
            Tuple[float, float]: (wind_vx, wind_vy) in m/s.
        """

        angle_rad = np.radians(self.wind_angle)
        wind_vx = self.wind_speed * np.cos(angle_rad)
        wind_vy = self.wind_speed * np.sin(angle_rad)
        return wind_vx, wind_vy

    def interpret_caliber(self):
        """
        Determines bullet diameter in meters based on caliber unit.

        - If caliber < 1.5: assumed inches -> converted to meters.
        - Otherwise: assumed millimeters -> converted to meters.

        Returns:
            float: Bullet diameter in meters.
        """

        if self.caliber < 1.5:
            return self.caliber * 0.0254  # 1 inch = 25.4 mm
        else:
            return self.caliber / 1000.0  # mm to m

    def get_cd_g7(self, mach):
        """
        Interpolates the drag coefficient (Cd) using G7 ballistic standard.

        Based on Mach number, performs linear interpolation between tabulated values.
        Accounts for aerodynamic characteristics at different speeds.

        Args:
            mach (float): Mach number (velocity / speed of sound).

        Returns:
            float: Interpolated drag coefficient.
        """

        g7_table = [
            (0.2, 0.150),
            (0.5, 0.145),
            (0.7, 0.140),
            (0.9, 0.200),
            (1.0, 0.250),
            (1.2, 0.300),
            (1.5, 0.220),
            (2.0, 0.180),
            (2.5, 0.160),
            (3.0, 0.140),
        ]

        # Poza zakresem — wartości skrajne
        if mach <= g7_table[0][0]:
            return g7_table[0][1]
        if mach >= g7_table[-1][0]:
            return g7_table[-1][1]

        # Interpolacja liniowa
        for i in range(1, len(g7_table)):
            m0, cd0 = g7_table[i - 1]
            m1, cd1 = g7_table[i]
            if mach <= m1:
                ratio = (mach - m0) / (m1 - m0)
                return cd0 + ratio * (cd1 - cd0)

        return g7_table[-1][1]  # fallback
  
    def simulate_trajectory(self, angle_rad, dt=0.002):
        """
        Simulates the bullet's flight trajectory with air drag, gravity, wind, and Coriolis force.

        Numerical integration (Euler method) is performed iteratively with:
        - Drag based on Cd, air density, and cross-sectional area.
        - Wind influence via velocity correction.
        - Gravity acting vertically.
        - Coriolis acceleration due to Earth's rotation.

        Args:
            angle_rad (float): Firing angle in radians.
            dt (float): Time step in seconds.

        Returns:
            np.ndarray: Array of trajectory points [t, x, y, z, velocity, kinetic_energy].
        """

        g = 9.81
        omega = 7.292115e-5
        azimuth = np.radians(self.azimuth_geo + self.azimuth_correction)
        phi = np.radians(self.lat)
        
        air_density = self.air_density
        mass_kg = self.mass_kg
        area = self.area
        speed_of_sound = self.speed_of_sound
        
        vx = self.velocity * np.cos(angle_rad) * np.cos(azimuth)
        vy = self.velocity * np.cos(angle_rad) * np.sin(azimuth)
        vz = self.velocity * np.sin(angle_rad)


        x, y, z, t = 0.0, 0.0, 0.0, 0.0
        points = [[t, x, y, z, self.velocity, 0.5 * self.mass_kg * self.velocity**2]]
        wind_vx, wind_vy = self.calculate_wind_components()

        extra_step = False
        while True:
            v_rel_x = vx - wind_vx
            v_rel_y = vy - wind_vy
            v_rel_z = vz
            v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

            mach = v_rel / speed_of_sound
            cd = self.get_cd_g7(mach)

            drag = 0.5 * air_density * v_rel**2 * cd * area
            factor = drag / (mass_kg * v_rel)
            
            ax = -factor * v_rel_x
            ay = -factor * v_rel_y
            az = -g - factor * v_rel_z

            # Coriolis effect
            a_coriolis = 2 * omega * np.sin(phi)
            a_coriolis_x = a_coriolis * vy * np.cos(azimuth)
            a_coriolis_y = -a_coriolis * vx * np.sin(azimuth)
            
            vx += (ax + a_coriolis_x) * dt
            vy += (ay + a_coriolis_y) * dt
            vz += az * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt
            t += dt

            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            energy = 0.5 * self.mass_kg * speed**2
            points.append([t, x, y, z, speed, energy])

            if x >= self.distance_to_target:
                if extra_step:
                    break
                extra_step = True

        traj = np.array(points)
        self.interpolated_results = {}  # Czyszczenie, jeśli metoda wywoływana wielokrotnie
        for distance in [self.distance_to_target, 100.0]:
            self.interpolated_results[distance] = self.interpolate_at_x(traj, distance)
        return traj

    def interpolate_at_x(self, traj, target_x):
        """
        Linearly interpolates trajectory data at a given horizontal position.

        Extracts intermediate data between two adjacent points:
            p(x) = p1 + ratio * (p2 - p1)

        Args:
            traj (np.ndarray): Simulated trajectory data.
            target_x (float): Desired horizontal position.

        Returns:
            np.ndarray: Interpolated trajectory point [t, x, y, z, v, E].
        """

        over_index = next((i for i, p in enumerate(traj) if p[1] >= target_x), None)
        if over_index is not None and over_index > 0:
            p1 = traj[over_index - 1]
            p2 = traj[over_index]
            denom = p2[1] - p1[1]
            if denom == 0:
                return [None, target_x, None, None, None, None]
            ratio = (target_x - p1[1]) / denom
            return p1 + ratio * (p2 - p1)
        else:
            return [None, target_x, None, None, None, None]

    def find_firing_angle(self, tolerance=0.001, max_steps=100):
        """
        Optimizes both the vertical firing angle and horizontal azimuth to ensure the bullet hits
        the target elevation and is centered laterally at the target distance.

        Uses binary search:
        - Outer loop: adjusts elevation angle (angle_rad) to match target height (Z)
        - Inner loop (optional): adjusts azimuth to minimize lateral offset (Y)

        Args:
            tolerance (float): Acceptable height error in meters.
            max_steps (int): Maximum iterations for convergence.

        Returns:
            float: Estimated firing angle in radians.
        """

        low = np.radians(0.01)
        high = np.radians(10.0)

        for _ in range(max_steps):
            angle = (low + high) / 2
            traj = self.simulate_trajectory(angle)
            z_at_target = self.interpolate_at_x(traj, self.distance_to_target)[3]

            if z_at_target is None:
                low = angle
                continue

            error = z_at_target - self.target_height

            if abs(error) < tolerance:
                return angle

            if error > 0:
                high = angle
            else:
                low = angle

            if abs(high - low) < 1e-8:
                break

        return angle

    def find_firing_azimuth(self, angle_rad, tolerance=0.001, max_steps=100):
        """
        Optimizes the horizontal firing angle (azimuth) to center the bullet laterally at the target.

        Uses binary search to minimize lateral deviation (Y) at the target distance.

        Args:
            angle_rad (float): Vertical firing angle (in radians).
            tolerance (float): Acceptable lateral error in meters.
            max_steps (int): Max search iterations.

        Returns:
            float: Optimal azimuth angle in degrees.
        """
        low = self.azimuth_correction - 5.0
        high = self.azimuth_correction + 5.0

        for _ in range(max_steps):
            firing_azimuth = (low + high) / 2
            self.azimuth_correction = firing_azimuth

            traj = self.simulate_trajectory(angle_rad)
            impact_point = self.interpolate_at_x(traj, self.distance_to_target)
            if impact_point is None:
                break

            y_at_target = impact_point[2]
            error = y_at_target

            if abs(error) < tolerance:
                return firing_azimuth

            if error > 0:
                high = firing_azimuth
            else:
                low = firing_azimuth

            if abs(high - low) < 1e-6:
                break

        return firing_azimuth


def save_to_csv(points, interpolated_results=None, filename="trajectory.csv"):
    """
    Saves the simulated trajectory and optional interpolation points to a CSV file.

    Args:
        points (list): List of trajectory data rows.
        interpolated_results (dict, optional): Additional data points (e.g. impact points).
        filename (str): Output CSV filename.
    """

    output = [list(p) for p in points]
    if interpolated_results:
        for key in sorted(interpolated_results.keys()):
            point = interpolated_results[key]
            if point[0] is not None:
                output.append(point.tolist()) if isinstance(point, np.ndarray) else point

    output.sort(key=lambda row: row[0])

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time [s]', 'X [m]', 'Y [m]', 'Z [m]', 'Speed [m/s]', 'Kinetic Energy [J]'])
        for row in output:
            writer.writerow([
                f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}",
                f"{row[3]:.3f}", f"{row[4]:.2f}", f"{row[5]:.1f}"
            ])

def save_to_png(traj, target_distance, target_height, filename="trajectory.png"):
    """
    Generates three plots:
    1. Vertical profile: height vs distance.
    2. Horizontal drift: lateral offset over distance.
    3. Time evolution: velocity and kinetic energy.

    Saves result as a PNG image.

    Args:
        traj (np.ndarray): Trajectory data.
        target_distance (float): Target horizontal position.
        target_height (float): Target vertical position.
        filename (str): Output PNG filename.
    """

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Wykres 1 — trajektoria pionowa (X vs Z)
    axs[0].plot(traj[:, 1], traj[:, 3], label="Z (height)")
    axs[0].plot(target_distance, target_height, 'rx', markersize=8, label='Target')
    axs[0].set_xlabel('Distance [m]')
    axs[0].set_ylabel('Height [m]')
    axs[0].set_title('Vertical trajectory (X-Z)')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlim(left=0)

    # Wykres 2 — boczny dryf (X vs Y)
    axs[1].plot(traj[:, 1], traj[:, 2], label="Y (lateral drift)")
    axs[1].plot(target_distance, 0, 'rx', markersize=8, label='Target line')
    axs[1].set_xlabel('Distance [m]')
    axs[1].set_ylabel('Side offset [m]')
    axs[1].set_title('Horizontal drift (X-Y)')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlim(left=0)

    # Wykres 3 — prędkość i energia kinetyczna w czasie (t vs V i E)
    ax3 = axs[2]
    time = traj[:, 0]
    speed = traj[:, 4]
    energy = traj[:, 5]

    ax3.plot(time, speed, 'b-', label='Velocity [m/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]', color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    ax3b = ax3.twinx()
    ax3b.plot(time, energy, 'g-', label='Kinetic Energy [J]')
    ax3b.set_ylabel('Energy [J]', color='g')
    ax3b.tick_params(axis='y', labelcolor='g')

    ax3.set_title('Velocity & Kinetic Energy over time')
    axs[2].set_xlim(left=0)
    
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    """
    Main execution block:
    - Parses command-line input
    - Constructs Bullet object with parameters
    - Simulates trajectory
    - Computes firing angle
    - Saves trajectory results to CSV and PNG
    - Displays impact info on console
    """

    def parse_args():
        """
        Parses command-line arguments using argparse.

        Supported options:
            --caliber, --bc, --mass_grain, --velocity, --distance_to_target,
            --target_height, --wind_speed, --wind_angle, --temp_c,
            --pressure_hpa, --humidity, --azimuth, --lat

        Returns:
            argparse.Namespace: Parsed user input as attributes.
        """

        parser = argparse.ArgumentParser(description="Ballistic trajectory simulator")
        parser.add_argument("--caliber", type=float, default=0.223, help="Caliber (inches or mm)")
        parser.add_argument("--bc", type=float, default=0.27, help="Ballistic coefficient (BC)")
        parser.add_argument("--mass_grain", type=float, default=55, help="Bullet mass in grains")
        parser.add_argument("--velocity", type=float, default=900, help="Initial velocity [m/s]")
        parser.add_argument("--distance_to_target", type=float, default=100, help="Target distance [m]")
        parser.add_argument("--target_height", type=float, default=0, help="Target height [m]")
        parser.add_argument("--wind_speed", type=float, default=0, help="Wind speed [m/s]")
        parser.add_argument("--wind_angle", type=float, default=0, help="Wind direction [°]")
        parser.add_argument("--temp_c", type=float, default=15, help="Temperature [°C]")
        parser.add_argument("--pressure_hpa", type=float, default=1013.25, help="Pressure [hPa]")
        parser.add_argument("--humidity", type=float, default=50, help="Humidity [%]")
        parser.add_argument("--azimuth", type=float, default=0, help="Azimuth [°]")
        parser.add_argument("--lat", type=float, default=52.0, help="Latitude [°]")
        return parser.parse_args()


    args = parse_args()
    bullet = Bullet(**vars(args))  # przekazujemy wszystkie jako keyword args

    traj = bullet.simulate_trajectory(bullet.angle_rad)
    impact = bullet.interpolated_impact

    save_to_csv(traj, bullet.interpolated_results)
    print("Results saved to trajectory.csv")
    print(f"Required firing angle: {np.degrees(bullet.angle_rad):.6f}°")
    print(f"Azimuth correction: {bullet.azimuth_correction:+.3f}° relative to target line")
    print(f"Total azimuth (absolute): {bullet.azimuth_geo + bullet.azimuth_correction:.3f}°")
    save_to_png(traj, bullet.distance_to_target, bullet.target_height)
