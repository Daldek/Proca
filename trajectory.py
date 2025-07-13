import numpy as np
import matplotlib.pyplot as plt
import csv

class Bullet:
    GRAIN_TO_KG = 0.00006479891
    DRAG_COEFF = 0.295
    BULLET_DIAMETER = 0.00782  # m (.308 cala)

    def __init__(self, mass_grain=55, velocity=900, distance_to_target=100, target_height=0,
                 temp_c=15, pressure_hpa=1013.25, humidity=50):
        self.mass_kg = mass_grain * self.GRAIN_TO_KG
        self.velocity = velocity
        self.distance_to_target = distance_to_target
        self.target_height = target_height
        self.area = np.pi * (self.BULLET_DIAMETER / 2) ** 2
        self.air_density = self.calculate_air_density(temp_c, pressure_hpa, humidity)
        self.interpolated_impact = None
        self.angle_rad = self.find_firing_angle()

    def calculate_air_density(self, temp_c, pressure_hpa, humidity):
        T = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        sat_pressure = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3)) * 100
        p_v = humidity / 100 * sat_pressure
        p_d = pressure_pa - p_v
        Rd = 287.058
        Rv = 461.495
        return (p_d / (Rd * T)) + (p_v / (Rv * T))

    def simulate_trajectory(self, angle_rad, dt=0.002):
        g = 9.81
        vx = self.velocity * np.cos(angle_rad)
        vy = self.velocity * np.sin(angle_rad)
        x, y, t = 0.0, 0.0, 0.0
        points = []

        extra_step = False
        while True:
            points.append([t, x, y])
            v = np.sqrt(vx ** 2 + vy ** 2)
            drag = 0.5 * self.air_density * self.DRAG_COEFF * self.area * v ** 2
            ax = -drag * vx / (v * self.mass_kg)
            ay = -g - (drag * vy / (v * self.mass_kg))
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt

            if x >= self.distance_to_target:
                if extra_step:
                    break
                extra_step = True

        traj = np.array(points)
        self.interpolated_impact = self.interpolate_at_x(traj, self.distance_to_target)
        self.interpolated_at_100m = self.interpolate_at_x(traj, 100.0) if self.distance_to_target >= 100 else None
        return traj

    def find_firing_angle(self, tolerance=0.001, max_steps=200):
        angle = np.radians(0.1)
        step = np.radians(0.1)
        last_direction = None

        for _ in range(max_steps):
            traj = self.simulate_trajectory(angle)
            over_index = next((i for i, p in enumerate(traj) if p[1] >= self.distance_to_target), None)
            if over_index is None or over_index == 0:
                angle += step
                continue

            p1 = traj[over_index - 1]
            p2 = traj[over_index]
            ratio = (self.distance_to_target - p1[1]) / (p2[1] - p1[1])
            y_at_target = p1[2] + ratio * (p2[2] - p1[2])
            error = abs(y_at_target - self.target_height)

            if error < tolerance:
                return angle

            direction = 1 if y_at_target < self.target_height else -1
            if last_direction is not None and direction != last_direction:
                step *= 0.5

            angle += direction * step
            angle = np.clip(angle, np.radians(0.01), np.radians(1.0))
            last_direction = direction

            if step < 1e-8:
                break

        return angle

    def interpolate_at_x(self, traj, target_x):
        over_index = next((i for i, p in enumerate(traj) if p[1] >= target_x), None)
        if over_index is not None and over_index > 0:
            p1 = traj[over_index - 1]
            p2 = traj[over_index]
            ratio = (target_x - p1[1]) / (p2[1] - p1[1])
            y = p1[2] + ratio * (p2[2] - p1[2])
            t = p1[0] + ratio * (p2[0] - p1[0])
            return [t, target_x, y]
        else:
            return [None, target_x, None]

def save_to_csv(points, filename, interpolated_point=None, interpolated_100m=None):
    output = [list(p) for p in points]
    
    if interpolated_point and interpolated_point[0] is not None:
        output.append(interpolated_point)
        if interpolated_100m and interpolated_100m[0] is not None:
            output.append(interpolated_100m)
    
    output.sort(key=lambda row: row[0])
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time [s]', 'X [m]', 'Y [m]'])
        for row in output:
            writer.writerow([f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}"])

def plot_trajectory(points, target_distance, target_height):
    plt.plot(points[:, 1], points[:, 2], label="Trajectory")
    plt.plot(target_distance, target_height, 'rx', markersize=8, label='Target')
    plt.xlabel('Distance [m]')
    plt.ylabel('Height [m]')
    plt.title('Bullet trajectory')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    try:
        mass = float(input("Enter bullet mass [grain] (default 55): ") or 55)
        velocity = float(input("Enter initial velocity [m/s] (default 900): ") or 900)
        distance = float(input("Enter target distance [m] (default 100): ") or 100)
        target_height = float(input("Enter target height [m] (default 0): ") or 0)
        temp_c = float(input("Enter temperature [°C] (default 15): ") or 15)
        pressure_hpa = float(input("Enter pressure [hPa] (default 1013.25): ") or 1013.25)
        humidity = float(input("Enter humidity [%] (default 50): ") or 50)
    except ValueError:
        mass, velocity, distance, target_height, temp_c, pressure_hpa, humidity = 55, 900, 100, 0, 15, 1013.25, 50

    bullet = Bullet(mass, velocity, distance, target_height,
                    temp_c, pressure_hpa, humidity)

    traj = bullet.simulate_trajectory(bullet.angle_rad)
    impact = bullet.interpolated_impact

    save_to_csv(traj, "trajectory.csv", bullet.interpolated_impact, bullet.interpolated_at_100m)
    print("Results saved to trajectory.csv")
    print(f"Required firing angle: {np.degrees(bullet.angle_rad):.6f}°")
    plot_trajectory(traj, distance, target_height)
