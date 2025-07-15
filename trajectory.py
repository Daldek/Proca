import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

class Atmosphere:
    """
    Represents environmental conditions for ballistics calculations.
    
    Attributes:
        wind_speed (float): Wind speed in m/s.
        wind_angle (float): Wind direction in degrees relative to X axis.
                            0° = headwind (against shot), 90° = wind from left,
                            180° = tailwind, 270° = wind from right.
        temp_c (float):
        pressure_hpa (float):
        humidity (float):
    """
    
    def __init__(self, wind_speed=0, wind_angle=0, temp_c=15, pressure_hpa=1013.25, humidity=50):
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.temp_c = temp_c
        self.pressure_hpa = pressure_hpa
        self.humidity = humidity
    
    @property
    def air_density(self):
        """
        Calculates the air density [kg/m³] based on ambient temperature, pressure, and humidity.

        Returns:
            float: Air density in kilograms per cubic meter.
        """
        t = self.temp_c + 273.15
        pressure_pa = self.pressure_hpa * 100
        sat_pressure = 6.1078 * 10 ** (7.5 * self.temp_c / (self.temp_c + 237.3)) * 100
        p_v = self.humidity / 100 * sat_pressure
        p_d = pressure_pa - p_v
        rd = 287.058
        rv = 461.495
        return (p_d / (rd * t)) + (p_v / (rv * t))

    @property
    def speed_of_sound(self):
        """
        Estimates the speed of sound [m/s] in humid air.

        Returns:
            float: Speed of sound in meters per second.
        """
        c = 331.3 * np.sqrt(1 + self.temp_c / 273.15)
        c += self.humidity * 0.1  # Approximate humidity correction: +0.1 m/s per 1% humidity
        return c

    @property
    def wind_components(self):
        """
        Resolves wind speed into X and Y components based on wind direction.

        Wind angle is measured relative to the shooter's firing line (X-axis):
        0° = headwind (against shot), 90° = wind from left,
        180° = tailwind, 270° = wind from right.

        Returns:
            tuple: (wind_vx, wind_vy) wind velocity components in m/s.
        """
        angle_rad = np.radians(self.wind_angle)
        wind_vx = self.wind_speed * np.cos(angle_rad)
        wind_vy = self.wind_speed * np.sin(angle_rad)
        return wind_vx, wind_vy


class Bullet:
    """
    Simulates a bullet's flight trajectory in 3D space with air drag, gravity, and wind influence.

    The shooter's firing line is aligned with the X-axis (shooter-target axis). The
    vertical angle controls elevation in the XZ plane, and an azimuth correction angle
    adjusts lateral offset in the XY plane to compensate wind drift.

    Attributes:
        caliber (float): Bullet diameter in inches or millimeters.
        bc (float): Ballistic coefficient indicating aerodynamic efficiency.
        mass_grain (float): Bullet mass in grains.
        velocity (float): Initial muzzle velocity in m/s.
        distance_to_target (float): Horizontal distance to target in meters.
        target_height (float): Target height relative to shooter in meters.
        wind_speed (float): Wind speed in m/s.
        wind_angle (float): Wind direction in degrees relative to X axis (0° = headwind).
        air_density (float): Air density in kg/m³ (calculated).
        speed_of_sound (float): Speed of sound in m/s (calculated).
        angle_rad (float): Computed vertical firing angle (radians).
        interpolated_results (dict): Interpolated trajectory points at specified distances.
    """

    GRAIN_TO_KG = 0.00006479891

    def __init__(self, caliber=0.223, bc=0.27, mass_grain=55, velocity=900, distance_to_target=100,
                 target_height=0, wind_speed=0, wind_angle=0, temp_c=15, pressure_hpa=1013.25,
                 humidity=50, shooter_coords=None, target_coords=None):
        self.caliber = caliber
        self.bc = bc
        self.mass_grain = mass_grain
        self.velocity = velocity
        self.distance_to_target = distance_to_target
        self.target_height = target_height
        self.atmosphere = Atmosphere(wind_speed, wind_angle, temp_c, pressure_hpa, humidity)
        self.shooter_coords = shooter_coords
        self.target_coords = target_coords
        
        if self.shooter_coords and self.target_coords:
            self.distance_to_target, _ = self.haversine_distance_and_azimuth
            
        self._bullet_diameter = self._interpret_caliber()
        self._bullet_area = np.pi * (self._bullet_diameter / 2) ** 2
        self._bullet_mass_kg = mass_grain * self.GRAIN_TO_KG
        self.angle_rad = self.find_firing_angle()
        self.interpolated_results = {}


    @property
    def haversine_distance_and_azimuth(self):
        """
        Calculates the great-circle distance and azimuth from shooter to target using geographical coordinates.

        Returns:
            tuple: (distance_km, azimuth_deg)
        """
        if self.target_coords is None:
            raise ValueError("Target coordinates not set.")

        lat1, lon1 = map(math.radians, self.shooter_coords)
        lat2, lon2 = map(math.radians, self.target_coords)

        # Difference in coordinates
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        # Haversine formula for distance
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        earth_radius = 6371000.0
        distance = earth_radius * c

        # Formula for azimuth (initial bearing)
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        azimuth_rad = math.atan2(x, y)
        azimuth_deg = (math.degrees(azimuth_rad) + 360) % 360
        return distance, azimuth_deg

    @property
    def bullet(self):
        return self._bullet_diameter, self._bullet_area, self._bullet_mass_kg
    
    def _interpret_caliber(self):
        """
        Converts caliber input to bullet diameter in meters.

        Assumes:
            - caliber < 1.5 interpreted as inches (converted to meters).
            - caliber >= 1.5 interpreted as millimeters (converted to meters).

        Returns:
            float: Bullet diameter in meters.
        """
        if self.caliber < 1.5:
            return self.caliber * 0.0254
        else:
            return self.caliber / 1000.0

    def get_cd_g7(self, mach):
        """
        Interpolates the drag coefficient (Cd) using the G7 ballistic standard table.

        Args:
            mach (float): Mach number (velocity / speed_of_sound).

        Returns:
            float: Drag coefficient.
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
        if mach <= g7_table[0][0]:
            return g7_table[0][1]
        if mach >= g7_table[-1][0]:
            return g7_table[-1][1]
        for i in range(1, len(g7_table)):
            m0, cd0 = g7_table[i - 1]
            m1, cd1 = g7_table[i]
            if mach <= m1:
                ratio = (mach - m0) / (m1 - m0)
                return cd0 + ratio * (cd1 - cd0)
        return g7_table[-1][1]

    def simulate_trajectory(self, angle_rad, azimuth_rad=0.0, dt=0.002):
        """
        Simulates the bullet's flight trajectory including air drag, gravity, and wind.

        Numerical integration is performed with Euler's method:
          - Drag depends on Cd, air density, and cross-sectional area.
          - Wind velocity modifies relative velocity.
          - Gravity acts vertically downward.
          - The firing azimuth adjusts lateral velocity to compensate wind drift.

        Args:
            angle_rad (float): Vertical firing angle in radians.
            azimuth_rad (float): Horizontal azimuth correction in radians (default 0).
            dt (float): Time step for simulation in seconds.

        Returns:
            np.ndarray: Array of trajectory points with columns:
                        [time, x, y, z, distance_xy, distance_3d, speed, kinetic_energy]
        """
        g = 9.81
        air_density = self.atmosphere.air_density
        mass_kg = self._bullet_mass_kg
        area = self._bullet_area
        speed_of_sound = self.atmosphere.speed_of_sound

        # Initial velocity components (accounting for azimuth correction)
        vx = self.velocity * np.cos(angle_rad) * np.cos(azimuth_rad)
        vy = self.velocity * np.cos(angle_rad) * np.sin(azimuth_rad)
        vz = self.velocity * np.sin(angle_rad)

        x, y, z, t = 0.0, 0.0, 0.0, 0.0
        points = [[t, x, y, z, 0.0, 0.0, self.velocity, 0.5 * self._bullet_mass_kg * self.velocity ** 2]]

        wind_vx, wind_vy = self.atmosphere.wind_components

        extra_step = False
        while True:
            v_rel_x = vx - wind_vx
            v_rel_y = vy - wind_vy
            v_rel_z = vz
            v_rel = np.sqrt(v_rel_x ** 2 + v_rel_y ** 2 + v_rel_z ** 2)

            mach = v_rel / speed_of_sound
            cd = self.get_cd_g7(mach)

            drag = 0.5 * air_density * v_rel ** 2 * cd * area
            factor = drag / (mass_kg * v_rel)

            ax = -factor * v_rel_x
            ay = -factor * v_rel_y
            az = -g - factor * v_rel_z

            vx += ax * dt
            vy += ay * dt
            vz += az * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt
            t += dt

            distance_xy = np.sqrt(x ** 2 + y ** 2)
            distance_3d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            energy = 0.5 * mass_kg * speed ** 2

            points.append([t, x, y, z, distance_xy, distance_3d, speed, energy])

            if x >= self.distance_to_target:
                if extra_step:
                    break
                extra_step = True

        traj = np.array(points)
        self.interpolated_results = {}
        for distance in [self.distance_to_target, 100.0]:
            self.interpolated_results[distance] = self.interpolate_at_x(traj, distance)

        return traj

    def interpolate_at_x(self, traj, target_x):
        """
        Linearly interpolates trajectory data at a given horizontal X position.

        Args:
            traj (np.ndarray): Trajectory data array.
            target_x (float): Desired X position for interpolation.

        Returns:
            np.ndarray: Interpolated trajectory point with columns:
                        [time, x, y, z, distance_xy, distance_3d, speed, kinetic_energy]
        """
        over_index = next((i for i, p in enumerate(traj) if p[1] >= target_x), None)
        if over_index is not None and over_index > 0:
            p1 = traj[over_index - 1]
            p2 = traj[over_index]
            denom = p2[1] - p1[1]
            if denom == 0:
                return np.array([None] * 8)
            ratio = (target_x - p1[1]) / denom
            interp = p1 + ratio * (p2 - p1)

            x, y, z = interp[1], interp[2], interp[3]
            distance_xy = np.sqrt(x ** 2 + y ** 2)
            distance_3d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            return np.append(interp, [distance_xy, distance_3d])
        else:
            return np.array([None] * 8)

    def find_firing_angle(self, tolerance=0.001, max_steps=100):
        """
        Uses binary search to find vertical firing angle that hits target height at target distance.

        Args:
            tolerance (float): Maximum acceptable error in height [m].
            max_steps (int): Maximum binary search iterations.

        Returns:
            float: Firing angle in radians.
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
        Finds the horizontal azimuth correction angle to minimize lateral wind drift.

        Uses binary search to find azimuth correction angle (radians) that minimizes lateral offset Y.

        Args:
            angle_rad (float): Fixed vertical firing angle (radians).
            tolerance (float): Acceptable lateral offset in meters.
            max_steps (int): Maximum iterations.

        Returns:
            float: Azimuth correction angle in radians.
        """
        low = -np.radians(5)
        high = np.radians(5)

        for _ in range(max_steps):
            azimuth = (low + high) / 2
            traj = self.simulate_trajectory(angle_rad, azimuth_rad=azimuth)
            y_at_target = self.interpolate_at_x(traj, self.distance_to_target)[2]

            if y_at_target is None:
                low = azimuth
                continue

            if abs(y_at_target) < tolerance:
                return azimuth

            if y_at_target > 0:
                high = azimuth
            else:
                low = azimuth

            if abs(high - low) < 1e-8:
                break

        return azimuth


def save_to_csv(points, interpolated_results=None, filename="trajectory.csv"):
    """
    Saves the simulated trajectory and optional interpolation points to a CSV file.

    Args:
        points (list): List of trajectory data rows.
        interpolated_results (dict, optional): Additional data points (e.g., impact points).
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
        writer.writerow(['Time [s]', 'X [m]', 'Y [m]', 'Z [m]', 'Distance XY [m]', 'Distance 3D [m]', 'Speed [m/s]', 'Kinetic Energy [J]'])
        for row in output:
            writer.writerow([
                f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}",
                f"{row[3]:.3f}", f"{row[4]:.3f}", f"{row[5]:.3f}",
                f"{row[6]:.3f}", f"{row[7]:.2f}"
            ])

def save_to_png(traj, target_distance, target_height, filename="trajectory.png"):
    fig = plt.figure(figsize=(10, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
    
    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax0.set_box_aspect([2, 1, 1])
    sc = ax0.scatter(traj[:, 1], traj[:, 2], traj[:, 3], c=traj[:, 6], cmap='plasma', s=5)
    ax0.scatter(target_distance, 0, target_height, color='red', label="Target", s=50)
    ax0.set_title("Isometric 3D view of trajectory")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")
    ax0.set_zlabel("Z [m]")
    ax0.view_init(elev=30, azim=330)
    fig.colorbar(sc, ax=ax0, shrink=0.6, label="Velocity [m/s]")
    ax0.legend()
    ax0.grid()
    ax0.tick_params(axis='x', pad=1)
    ax0.tick_params(axis='y', pad=1)
    for label in ax0.get_yticklabels():
        label.set_rotation(330)


    axs = [fig.add_subplot(gs[i]) for i in range(1, 4)]

    # 1. Trajektoria pionowa
    axs[0].plot(traj[:, 4], traj[:, 3], label="Z (height)")
    axs[0].plot(target_distance, target_height, 'rx', markersize=8, label='Target')
    axs[0].set_title('Vertical trajectory (X-Z)')
    axs[0].set_xlabel('Distance (X) [m]')
    axs[0].set_ylabel('Height (Z) [m]')
    axs[0].legend()
    axs[0].grid()

    # 2. Dryf boczny
    axs[1].plot(traj[:, 4], traj[:, 2], label="Y (lateral drift)")
    axs[1].plot(target_distance, 0, 'rx', markersize=8, label='Target line')
    axs[1].set_title('Horizontal drift (X-Y)')
    axs[1].set_xlabel('Distance (X) [m]')
    axs[1].set_ylabel('Side offset (Y) [m]')
    axs[1].legend()
    axs[1].grid()

    # 3. Prędkość i energia
    time = traj[:, 0]
    speed = traj[:, 6]
    energy = traj[:, 7]
    ax3 = axs[2]
    ax3.plot(time, speed, 'b-', label='Velocity [m/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3b = ax3.twinx()
    ax3b.plot(time, energy, 'g-', label='Kinetic Energy [J]')
    ax3b.set_ylabel('Energy [J]', color='g')
    ax3b.tick_params(axis='y', labelcolor='g')
    ax3.set_title('Velocity & Kinetic Energy over time')

    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

if __name__ == "__main__":
    def parse_args():
        """
        Parses command-line arguments for ballistic trajectory simulation.

        Returns:
            argparse.Namespace: Parsed arguments.
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
        return parser.parse_args()

    args = parse_args()
    bullet = Bullet(**vars(args))

    # Find azimuth correction to minimize lateral offset (wind effect)
    azimuth_correction = bullet.find_firing_azimuth(bullet.angle_rad)
    traj = bullet.simulate_trajectory(bullet.angle_rad, azimuth_rad=azimuth_correction)

    save_to_csv(traj, bullet.interpolated_results)
    print("Results saved to trajectory.csv")
    print(f"Required firing angle: {np.degrees(bullet.angle_rad):.6f}°")
    print(f"Azimuth correction: {np.degrees(azimuth_correction):+.3f}° relative to shooter-target axis")
    print(f"Effective firing azimuth: {np.degrees(azimuth_correction):.3f}°")
    save_to_png(traj, bullet.distance_to_target, bullet.target_height)
