import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

class Bullet:
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
        self.azimuth = azimuth
        self.lat = lat
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

    def calculate_speed_of_sound(self, temp_c, humidity):
        # Podstawowy wzór z temperaturą
        c = 331.3 * np.sqrt(1 + temp_c / 273.15)

        # Opcjonalne poprawki na wilgotność – uproszczone (niewielki wpływ)
        c += humidity * 0.1  # szacunkowo: +0.1 m/s na każdy 1% wilgotności

        return c

    def calculate_wind_components(self):
        angle_rad = np.radians(self.wind_angle)
        wind_vx = self.wind_speed * np.cos(angle_rad)
        wind_vy = self.wind_speed * np.sin(angle_rad)
        return wind_vx, wind_vy

    def interpret_caliber(self):
        if self.caliber < 1.5:
            # prawdopodobnie cale
            return self.caliber * 0.0254  # 1 cal = 25.4 mm
        else:
            # prawdopodobnie milimetry
            return self.caliber / 1000.0  # milimetry na metry

    def get_cd_g7(self, mach):
        # Prosta interpolacja liniowa pomiędzy punktami
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
        g = 9.81
        omega = 7.292115e-5
        azimuth = np.radians(self.azimuth)
        phi = np.radians(self.lat)
        vx = self.velocity * np.cos(angle_rad)  # kierunek do przodu (X)
        vz = self.velocity * np.sin(angle_rad)  # wysokość (Z)
        vy = 0.0  # boczny dryf (Y)

        x, y, z, t = 0.0, 0.0, 0.0, 0.0
        points = []
        wind_vx, wind_vy = self.calculate_wind_components()

        extra_step = False
        while True:
            v_rel_x = vx - wind_vx
            v_rel_y = vy - wind_vy
            v_rel_z = vz
            v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

            mach = v_rel / self.speed_of_sound
            cd = self.get_cd_g7(mach)

            drag = 0.5 * self.air_density * v_rel**2 * cd * self.area
            ax = -drag * v_rel_x / (v_rel * self.mass_kg)
            ay = -drag * v_rel_y / (v_rel * self.mass_kg)
            az = -g - drag * v_rel_z / (v_rel * self.mass_kg)

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
        self.interpolated_impact = self.interpolate_at_x(traj, self.distance_to_target)
        self.interpolated_at_100m = self.interpolate_at_x(traj, 100.0) if self.distance_to_target > 100 else None
        return traj

    def interpolate_at_x(self, traj, target_x):
        over_index = next((i for i, p in enumerate(traj) if p[1] >= target_x), None)
        if over_index is not None and over_index > 0:
            p1 = traj[over_index - 1]
            p2 = traj[over_index]
            denom = p2[1] - p1[1]
            if denom == 0:
                return [None, target_x, None, None, None, None]
            ratio = (target_x - p1[1]) / (p2[1] - p1[1])
            t = p1[0] + ratio * (p2[0] - p1[0])
            y = p1[2] + ratio * (p2[2] - p1[2])
            z = p1[3] + ratio * (p2[3] - p1[3])
            speed = p1[4] + ratio * (p2[4] - p1[4])
            energy = p1[5] + ratio * (p2[5] - p1[5])
            return [t, target_x, y, z, speed, energy]
        else:
            return [None, target_x, None, None, None, None]

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
            z_at_target = p1[3] + ratio * (p2[3] - p1[3])
            error = abs(z_at_target - self.target_height)

            if error < tolerance:
                return angle

            direction = 1 if z_at_target < self.target_height else -1
            if last_direction is not None and direction != last_direction:
                step *= 0.5

            angle += direction * step
            angle = np.clip(angle, np.radians(0.01), np.radians(10.0))
            last_direction = direction

            if step < 1e-8:
                break

        return angle

def save_to_csv(points, filename, interpolated_point=None, interpolated_100m=None):
    output = [list(p) for p in points]
    if interpolated_point and interpolated_point[0] is not None:
        output.append(interpolated_point)
    if interpolated_100m and interpolated_100m[0] is not None:
        output.append(interpolated_100m)

    output.sort(key=lambda row: row[0])

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time [s]', 'X [m]', 'Y [m]', 'Z [m]', 'Speed [m/s]', 'Kinetic Energy [J]'])
        for row in output:
            writer.writerow([
                f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}",
                f"{row[3]:.3f}", f"{row[4]:.2f}", f"{row[5]:.1f}"
            ])

def plot_all(traj, target_distance, target_height):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Wykres 1 — trajektoria pionowa (X vs Z)
    axs[0].plot(traj[:, 1], traj[:, 3], label="Z (height)")
    axs[0].plot(target_distance, target_height, 'rx', markersize=8, label='Target')
    axs[0].set_xlabel('Distance [m]')
    axs[0].set_ylabel('Height [m]')
    axs[0].set_title('Vertical trajectory (X-Z)')
    axs[0].legend()
    axs[0].grid()

    # Wykres 2 — boczny dryf (X vs Y)
    axs[1].plot(traj[:, 1], traj[:, 2], label="Y (lateral drift)")
    axs[1].plot(target_distance, 0, 'rx', markersize=8, label='Target line')
    axs[1].set_xlabel('Distance [m]')
    axs[1].set_ylabel('Side offset [m]')
    axs[1].set_title('Horizontal drift (X-Y)')
    axs[1].legend()
    axs[1].grid()

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
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    def parse_args():
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

    save_to_csv(traj, "trajectory.csv", bullet.interpolated_impact, bullet.interpolated_at_100m)
    print("Results saved to trajectory.csv")
    print(f"Required firing angle: {np.degrees(bullet.angle_rad):.6f}°")
    plot_all(traj, bullet.distance_to_target, bullet.target_height)
