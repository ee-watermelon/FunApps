

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from ahrs.filters import Madgwick, madgwick
from vpython import box, color, compound, cylinder, rate, scene, vector


GRAVITY = 9.80665
MODE_NAMES = {0: "stationary", 1: "tilt", 2: "rotate", 3: "shake", 4: "combined"}
feed = np.array([1.0, 0.0, 0.0, 0.0])
madg = Madgwick(frequency=50.0)
board = box(length=4, height=0.2, width=2, color=color.cyan)
pointer = cylinder(pos=vector(0, 0.1, 0), axis=vector(1.5, 0, 0), radius=0.1, color=color.red)
imu_model = compound([board, pointer])


class VirtualIMU:
   

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.mode = 1
        self.gyro_bias = np.array([0.18, -0.12, 0.22])  # degrees/second

    def _orientation(self, t: float) -> np.ndarray:
     
        if self.mode == 0:
            return np.zeros(3)
        if self.mode == 1:
            return np.array([25 * math.sin(0.7 * t), 18 * math.sin(0.45 * t), 0])
        if self.mode == 2:
            return np.array([0, 0, (45 * t) % 360])
        if self.mode == 3:
            return np.array([4 * math.sin(8 * t), 3 * math.sin(11 * t), 2 * math.sin(9 * t)])
        return np.array([
            25 * math.sin(0.7 * t),
            18 * math.sin(0.45 * t),
            (25 * t) % 360,
        ])

    def _linear_acceleration(self, t: float) -> np.ndarray:
        if self.mode == 3:
            return np.array([4 * math.sin(13 * t), 3 * math.sin(17 * t), 2 * math.sin(11 * t)])
        if self.mode == 4:
            return np.array([0.8 * math.sin(2 * t), 0.5 * math.sin(3 * t), 0.3 * math.sin(4 * t)])
        return np.zeros(3)

    @staticmethod
    def _gravity_in_body(angles_deg: np.ndarray) -> np.ndarray:
        roll, pitch, _ = np.radians(angles_deg)
        # Accelerometer reading caused by gravity, expressed in the sensor frame.
        return GRAVITY * np.array([
            -math.sin(pitch),
            math.sin(roll) * math.cos(pitch),
            math.cos(roll) * math.cos(pitch),
        ])

    def read(self, t: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = self._orientation(t)
        previous = self._orientation(max(0.0, t - dt))
        gyro = (angles - previous) / dt
        # Correct the artificial wraparound of yaw at 360 degrees.
        gyro[2] = ((angles[2] - previous[2] + 180) % 360 - 180) / dt

        accel = self._gravity_in_body(angles) + self._linear_acceleration(t)
        accel += self.rng.normal(0, 0.055, 3)       # m/s^2 noise
        gyro += self.gyro_bias + self.rng.normal(0, 0.08, 3)
        return accel, gyro, angles


class ComplementaryFilter:
    

    def __init__(self, alpha: float = 0.98) -> None:
        self.alpha = alpha
        self.angles = np.zeros(3)

    def reset(self) -> None:
        self.angles[:] = 0

    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        ax, ay, az = accel
        accel_roll = math.degrees(math.atan2(ay, math.sqrt(ax * ax + az * az)))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

        predicted = self.angles + gyro * dt
        self.angles[0] = self.alpha * predicted[0] + (1 - self.alpha) * accel_roll
        self.angles[1] = self.alpha * predicted[1] + (1 - self.alpha) * accel_pitch
        self.angles[2] = predicted[2]  # yaw needs a magnetometer to prevent long-term drift
        return self.angles.copy()


def run_console(duration: float, sample_rate: float, output: Path | None) -> None:
    
    imu = VirtualIMU()
    filt = ComplementaryFilter()
    dt = 1.0 / sample_rate
    rows = []
    for index in range(round(duration * sample_rate)):
        t = index * dt
        accel, gyro, truth = imu.read(t, dt)
        estimate = filt.update(accel, gyro, dt)
        rows.append([t, *accel, *gyro, *truth, *estimate])

    if output:
        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "time_s", "ax_mps2", "ay_mps2", "az_mps2",
                "gx_dps", "gy_dps", "gz_dps",
                "true_roll_deg", "true_pitch_deg", "true_yaw_deg",
                "estimated_roll_deg", "estimated_pitch_deg", "estimated_yaw_deg",
            ])
            writer.writerows(rows)
        print(f"Saved {len(rows)} simulated samples to {output}")
    else:
        print("Last simulated sample:")
        print(np.array(rows[-1]))


def run_live(sample_rate: float) -> None:
    imu = VirtualIMU()
    filt = ComplementaryFilter()
    dt = 1.0 / sample_rate
    window_seconds = 10
    capacity = round(window_seconds * sample_rate)
    history = {name: deque(maxlen=capacity) for name in (
        "t", "ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "true_roll", "true_pitch"
    )}
    start = time.perf_counter()

    fig, (ax_sensor, ax_angle) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.canvas.manager.set_window_title("Virtual IMU")
    fig.suptitle("Virtual IMU mode: tilt")

    sensor_lines = []
    for key, label in (("ax", "Ax"), ("ay", "Ay"), ("az", "Az")):
        line, = ax_sensor.plot([], [], label=f"{label}")
        sensor_lines.append((key, line))
    ax_sensor.set_ylabel("Acceleration")
    ax_sensor.set_ylim(-15, 15)
    ax_sensor.grid(True, alpha=0.3)
    ax_sensor.legend(loc="upper left", ncol=3)

    angle_lines = []
    styles = (("roll", "Estimated roll", "-"), ("true_roll", "True roll", "--"),
              ("pitch", "Estimated pitch", "-"), ("true_pitch", "True pitch", "--"))
    for key, label, style in styles:
        line, = ax_angle.plot([], [], style, label=label)
        angle_lines.append((key, line))
    ax_angle.set_xlabel("Time (seconds)")
    ax_angle.set_ylabel("Angle (degrees)")
    ax_angle.set_ylim(-45, 45)
    ax_angle.grid(True, alpha=0.3)
    ax_angle.legend(loc="upper left", ncol=2)
    fig.text(0.5, 0.01, "Keys: 0 stationary | 1 tilt | 2 rotate | 3 shake | 4 combined | R reset | Q quit",
             ha="center")

    def on_key(event) -> None:
        if event.key in {"0", "1", "2", "3", "4"}:
            imu.mode = int(event.key)
            filt.reset()
            fig.suptitle(f"Virtual IMU mode: {MODE_NAMES[imu.mode]}")
        elif event.key == "r":
            filt.reset()
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def quaternion_to_rotation_matrix(feed):
      w, x, y, z = feed
      return np.array([
          [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
          [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
          [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
      ])

    def update_3D(gyro_in, accel_in):
      global feed
      feed = madg.updateIMU(feed, gyro_in, accel_in)
      R = quaternion_to_rotation_matrix(feed)
      imu_model.axis = vector(R[0,0], R[1,0], R[2,0])
      imu_model.up = vector(R[0,1], R[1,1], R[2,1])
    
    t = 0.0
    for gy0 in range(100):
      for gy1 in range(100):
        for gy2 in range(100):
          accel, gyro, truth = imu.read(t, dt)
             
          gyro[0] += (0.01*gy0)
          gyro[1] += (0.01*gy1)
          gyro[2] += (0.01*gy2)
          rate(10)
          update_3D(gyro, accel)
          t += 0.0002


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a six-axis IMU without hardware.")
    parser.add_argument("--rate", type=float, default=50, help="sample rate in Hz (default: 50)")
    parser.add_argument("--duration", type=float, default=10, help="console simulation duration")
    parser.add_argument("--csv", type=Path, help="run without GUI and save simulated samples to CSV")
    parser.add_argument("--no-gui", action="store_true", help="run a console-only simulation")
    args = parser.parse_args()
    if args.rate <= 0 or args.duration <= 0:
        parser.error("--rate and --duration must be positive")
    if args.csv or args.no_gui:
        run_console(args.duration, args.rate, args.csv)
    else:
        run_live(args.rate)


if __name__ == "__main__":
    main()

