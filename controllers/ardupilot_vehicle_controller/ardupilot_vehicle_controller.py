#!/usr/bin/env python3
'''
General ardupilot vehicle controller for Webots 2023a

AP_FLAKE8_CLEAN
'''


import webots_vehicle
import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from webots_vehicle import WebotsArduVehicle
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--motors", "-m",
                        type=str,
                        default="motor1, motor2, motor3, motor4",
                        help="Comma spaced list of motor names in ardupilot numerical order (ex --motors \"m1,m2,m3, m4\")")
    parser.add_argument("--servos", "-s",
                        type=str,
                        default="camera_tilt",
                        help="Comma spaced list of servo names, starting for SERVO7")
    parser.add_argument("--reversed-motors", "-r",
                        type=str,
                        default="3,4",
                        help="Comma spaced list of motors to reverse (starting from 1, in ardupilot order)")
    parser.add_argument("--bidirectional-motors",
                        type=bool,
                        default=False,
                        help="If the motors are bidirectional (as is the case for Rovers usually)")
    parser.add_argument("--uses-propellers",
                        type=bool,
                        default=True,
                        help="Whether the vehicle uses propellers. This is important as we need to linearize thrust if so")
    parser.add_argument("--motor-cap",
                        type=float,
                        default=float('inf'),
                        help="Motor velocity cap. This is useful for the crazyflie which default has way too much power")

    parser.add_argument("--accel",
                        type=str,
                        default="accelerometer",
                        help="Webots accelerometer name")
    parser.add_argument("--imu",
                        type=str,
                        default="inertial unit",
                        help="Webots IMU name")
    parser.add_argument("--gyro",
                        type=str,
                        default="gyro",
                        help="Webots gyro name")
    parser.add_argument("--gps",
                        type=str,
                        default="gps",
                        help="Webots GPS name")

    parser.add_argument("--camera",
                        type=str,
                        default=None,
                        help="Webots Camera name (optional)")
    parser.add_argument("--camera-fps",
                        type=int,
                        default=10,
                        help="Camera FPS. Note lower FPS is faster")
    parser.add_argument("--camera-port",
                        type=int,
                        default=None,
                        help="Port to stream grayscale camera images to. "
                             "If no port is supplied the camera will not be streamed.")

    parser.add_argument("--rangefinder",
                        type=str,
                        default=None,
                        help="Webots RangeFinder name (optional)")
    parser.add_argument("--rangefinder-fps",
                        type=int,
                        default=10,
                        help="rangefinder FPS. Note lower FPS is faster")
    parser.add_argument("--rangefinder-port",
                        type=int,
                        default=None,
                        help="Port to stream grayscale rangefinder images to. "
                             "If no port is supplied the rangefinder will not be streamed.")

    parser.add_argument("--instance", "-i",
                        type=int,
                        default=0,
                        help="Drone instance to match the SITL. This allows multiple vehicles")
    parser.add_argument("--sitl-address",
                        type=str,
                        default="127.0.0.1",
                        help="IP address of the SITL (useful with WSL2 eg \"172.24.220.98\")")

    return parser.parse_args()

def save_telemetry_plots(data: dict):
    if not data['time']:
        print("Nenhum dado de telemetria registrado.")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "metricas", "plots"))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"sensores_telemetria_{timestamp}.png"
    save_path = os.path.join(output_dir, file_name)

    t = np.array(data['time'])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f'Telemetria dos Sensores ({timestamp})', fontsize=14, fontweight='bold')

    # 1. Acelerômetro
    axes[0].plot(t, data['accel_x'], label='Acc X ($m/s^2$)', color='crimson')
    axes[0].plot(t, data['accel_y'], label='Acc Y ($m/s^2$)', color='forestgreen')
    axes[0].plot(t, data['accel_z'], label='Acc Z ($m/s^2$)', color='royalblue')
    axes[0].set_ylabel('Aceleração ($m/s^2$)')
    axes[0].set_title('Acelerômetro')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # 2. IMU
    axes[1].plot(t, np.degrees(data['roll']), label='Roll (°)', color='coral')
    axes[1].plot(t, np.degrees(data['pitch']), label='Pitch (°)', color='teal')
    axes[1].plot(t, np.degrees(data['yaw']), label='Yaw (°)', color='purple')
    axes[1].set_ylabel('Ângulo (°)')
    axes[1].set_title('IMU (Atitude)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    # 3. GPS
    axes[2].plot(t, data['gps_x'], label='Pos X (m)', color='darkorange')
    axes[2].plot(t, data['gps_y'], label='Pos Y (m)', color='navy')
    axes[2].plot(t, data['gps_z'], label='Pos Z / Altura (m)', color='darkgreen')
    axes[2].set_xlabel('Tempo de Simulação - robot.getTime() (s)')
    axes[2].set_ylabel('Posição (m)')
    axes[2].set_title('GPS (Posição Relativa)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right')

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_telemetry_csv(data: dict):
    if not data['time']:
        print("Nenhum dado de telemetria para salvar no CSV.")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "metricas", "csv"))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sensores_telemetria_{timestamp}.csv")

    fieldnames = [
        'time',
        'accel_x', 'accel_y', 'accel_z',
        'roll_deg', 'pitch_deg', 'yaw_deg',
        'gps_x', 'gps_y', 'gps_z'
    ]

    # conversao angulo --> graus
    roll_deg = np.degrees(data['roll'])
    pitch_deg = np.degrees(data['pitch'])
    yaw_deg = np.degrees(data['yaw'])

    num_samples = len(data['time'])

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_samples):
            writer.writerow({
                'time': data['time'][i],
                'accel_x': data['accel_x'][i],
                'accel_y': data['accel_y'][i],
                'accel_z': data['accel_z'][i],
                'roll_deg': roll_deg[i],
                'pitch_deg': pitch_deg[i],
                'yaw_deg': yaw_deg[i],
                'gps_x': data['gps_x'][i],
                'gps_y': data['gps_y'][i],
                'gps_z': data['gps_z'][i],
            })


if __name__ == "__main__":
    args = get_args()

    # parse string arguments into lists
    motors = [x.strip() for x in args.motors.split(',')]
    servos = [x.strip() for x in args.servos.split(',')]
    if args.reversed_motors:
        reversed_motors = [int(x) for x in args.reversed_motors.split(",")]
    else:
        reversed_motors = []

    vehicle = WebotsArduVehicle(motor_names=motors,
                                servo_names=servos,
                                reversed_motors=reversed_motors,
                                accel_name=args.accel,
                                imu_name=args.imu,
                                gyro_name=args.gyro,
                                gps_name=args.gps,
                                camera_name=args.camera,
                                camera_fps=args.camera_fps,
                                camera_stream_port=args.camera_port,
                                rangefinder_name=args.rangefinder,
                                rangefinder_fps=args.rangefinder_fps,
                                rangefinder_stream_port=args.rangefinder_port,
                                instance=args.instance,
                                motor_velocity_cap=args.motor_cap,
                                bidirectional_motors=args.bidirectional_motors,
                                uses_propellers=args.uses_propellers,
                                sitl_address=args.sitl_address)

    # User code (ex: connect via drone kit and take off)
    # ...

    try:
        while vehicle.webots_connected():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupção manual detectada. Gerando gráficos...")
    finally:
        timestamp_sessao = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_telemetry_csv(vehicle.telemetry)
        save_telemetry_plots(vehicle.telemetry)