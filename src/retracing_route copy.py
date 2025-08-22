import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import read_file
from detect_steps import detect_steps_threshold, apply_lowpass_filter
from estimate_step_length import extract_step_accelerations, weinberg_step_length

def moving_average(data, window_size=5):
    """Apply moving average smoothing to data."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

class PedestrianDeadReckoning:
    def __init__(self, initial_position=(0, 0), initial_heading=0):
        """
        Initialize PDR system.
        
        Args:
            initial_position: (x, y) starting position in meters
            initial_heading: initial heading in degrees (0 = North, 90 = East)
        """
        self.position = np.array(initial_position, dtype=float)
        self.heading = initial_heading
        self.path = [self.position.copy()]
        self.headings = [self.heading]
        self.mag_calibration_matrix = np.eye(3)  # Default: no calibration
        self.mag_offset = np.zeros(3)            # Default: no offset
        
    def calibrate_magnetometer(self, mx_data, my_data, mz_data):
        """
        Simple magnetometer calibration to remove hard iron offsets.
        For chest-mounted device.
        
        Args:
            mx_data, my_data, mz_data: Arrays of magnetometer readings
            
        Returns:
            calibration_matrix, offset: Calibration parameters
        """
        # Calculate offsets (hard iron correction)
        mx_offset = (np.max(mx_data) + np.min(mx_data)) / 2
        my_offset = (np.max(my_data) + np.min(my_data)) / 2
        mz_offset = (np.max(mz_data) + np.min(mz_data)) / 2
        
        self.mag_offset = np.array([mx_offset, my_offset, mz_offset])
        print(f"Magnetometer calibration offsets: {self.mag_offset}")
        
        # For now, we'll use simple offset correction
        # A full ellipsoid fitting would be better for production
        return np.eye(3), np.array([mx_offset, my_offset, mz_offset])
        
    def chest_mounted_to_earth_frame(self, ax, ay, az, mx, my, mz):
        """
        Transform sensor data from chest-mounted phone (screen facing outward)
        to Earth frame coordinates.
        
        For chest-mounted orientation:
        - Z-axis of phone points forward (away from chest - direction of walking)
        - Y-axis points downward along body
        - X-axis points to the right side
        
        Args:
            ax, ay, az: Acceleration in phone frame
            mx, my, mz: Magnetometer readings in phone frame
            
        Returns:
            a_north, a_east, a_down: Acceleration in Earth frame
            heading_deg: Heading in degrees (0=North, 90=East)
        """
        # Apply magnetometer calibration
        mx_cal = mx - self.mag_offset[0]
        my_cal = my - self.mag_offset[1]
        mz_cal = mz - self.mag_offset[2]
        
        # Step 1: Estimate gravity direction from accelerometer 
        acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if acc_norm < 0.1:  # Avoid division by zero
            return 0, 0, 0, 0
            
        # Normalize acceleration to get unit gravity vector
        gx, gy, gz = ax/acc_norm, ay/acc_norm, az/acc_norm
        
        # Step 2: Calculate device tilt (pitch and roll)
        # For chest-mounted orientation with screen facing outward:
        # - Pitch (forward lean) is rotation around X-axis 
        # - Roll (side lean) is rotation around Z-axis
        pitch = np.arctan2(-gy, np.sqrt(gx**2 + gz**2))  # Forward/backward tilt
        roll = np.arctan2(gx, gz)  # Side tilt
        
        # Step 3: Tilt-compensated magnetometer reading
        # Rotate magnetometer readings to horizontal plane
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        
        # Tilt compensation - this will align the magnetometer readings to the horizontal plane
        # For chest-mounted device with screen facing outward:
        mx_comp = (mx_cal * cos_pitch) + (mz_cal * sin_pitch)
        my_comp = (mx_cal * sin_roll * sin_pitch) + (my_cal * cos_roll) - (mz_cal * sin_roll * cos_pitch)
        mz_comp = (-mx_cal * sin_pitch) + (mz_cal * cos_pitch)  # Added this line to define mz_comp
        
        # Step 4: Calculate heading from compensated magnetometer
        # For chest-mount with screen facing outward:
        heading_rad = np.arctan2(mx_comp, mz_comp)
        heading_deg = np.degrees(heading_rad)
        
        # Normalize to 0-360 degrees
        heading_deg = (heading_deg + 360) % 360
        
        # Step 5: Transform acceleration to Earth frame
        # Create rotation matrix from pitch, roll, and heading
        # For chest-mounted device, we need to account for the 90-degree rotation around X-axis
        chest_to_body_rotation = Rotation.from_euler('x', 90, degrees=True)
        orientation_rotation = Rotation.from_euler('zyx', [heading_rad, roll, pitch])
        
        # Combined rotation: first from chest to body frame, then apply orientation
        total_rotation = orientation_rotation * chest_to_body_rotation
        
        # Apply rotation to acceleration vector
        acc_phone = np.array([ax, ay, az])
        acc_earth = total_rotation.apply(acc_phone)
        
        # Return north, east, down components and heading
        return acc_earth[0], acc_earth[1], acc_earth[2], heading_deg
    
    def update_position(self, step_length, direction_deg):
        """
        Update position based on step length and direction.
        
        Args:
            step_length: length of step in meters
            direction_deg: direction in degrees (0 = North, 90 = East)
        """
        direction_rad = np.radians(direction_deg)
        
        # Calculate displacement
        dx = step_length * np.sin(direction_rad)  # East displacement
        dy = step_length * np.cos(direction_rad)  # North displacement
        
        # Update position
        self.position[0] += dx  # x = East
        self.position[1] += dy  # y = North
        
        # Store in path
        self.path.append(self.position.copy())
        self.headings.append(direction_deg)
    
    def process_sensor_data(self, df, K=0.5, window_size=10):
        """
        Process complete sensor data to retrace route.
        
        Args:
            df: DataFrame with sensor data
            K: Weinberg calibration constant
            window_size: window size for step detection
        
        Returns:
            path: array of (x, y) positions
            headings: array of headings at each step
        """
        print("=== Processing Sensor Data for PDR ===")
        
        # Extract sensor data
        timestamps = df['timestamp'].values.astype(float)
        ax = df['ax'].values.astype(float)
        ay = df['ay'].values.astype(float)
        az = df['az'].values.astype(float)
        mx = df['mx'].values.astype(float)
        my = df['my'].values.astype(float)
        mz = df['mz'].values.astype(float)
        svm_data = df['svm'].values
        
        print(f"Data points: {len(df)}")
        
        # Calibrate magnetometer with available data
        self.calibrate_magnetometer(mx, my, mz)
        
        # Apply low-pass filter and detect steps
        print("=== Detecting Steps ===")
        filtered_svm = apply_lowpass_filter(svm_data, cutoff_freq=15.0, sampling_rate=50.0)
        adaptive_threshold = np.mean(filtered_svm) + 0.5 * np.std(filtered_svm)
        step_count, step_times, step_indices = detect_steps_threshold(
            filtered_svm, timestamps, adaptive_threshold, min_time_between_steps=0.3
        )
        
        print(f"Steps detected: {step_count}")
        
        if step_count == 0:
            print("No steps detected. Cannot trace route.")
            return np.array(self.path), np.array(self.headings)
        
        # Extract step accelerations for length estimation
        step_accelerations = extract_step_accelerations(svm_data, step_indices, window_size)
        step_lengths = weinberg_step_length(
            step_accelerations['a_max'], 
            step_accelerations['a_min'], 
            K
        )
        
        # Process and smooth headings before applying them
        all_headings = []
        for i in range(len(df)):
            _, _, _, heading = self.chest_mounted_to_earth_frame(
                ax[i], ay[i], az[i], mx[i], my[i], mz[i]
            )
            all_headings.append(heading)
            
        # Smooth headings using moving average
        if len(all_headings) >= 5:  # Need at least 5 points for a window of 5
            smoothed_headings = moving_average(all_headings, 5)
            # Pad the beginning to match original length
            pad_length = len(all_headings) - len(smoothed_headings)
            smoothed_headings = np.pad(smoothed_headings, (pad_length, 0), 'edge')
        else:
            smoothed_headings = all_headings
        
        print(f"Average step length: {np.mean(step_lengths):.3f} m")
        
        # Process each step
        print("=== Tracing Route with Chest-Mounted Orientation ===")
        for i, step_idx in enumerate(step_indices):
            # Get smoothed heading at step time
            step_heading = smoothed_headings[step_idx]
            
            # Update position using the step length and smoothed direction
            self.update_position(step_lengths[i], step_heading)
            
            if i < 10 or i % 20 == 0:  # Print first 10 steps and every 20th step
                print(f"Step {i+1}: length={step_lengths[i]:.3f}m, "
                      f"heading={step_heading:.1f}°, "
                      f"pos=({self.position[0]:.2f}, {self.position[1]:.2f})")
        
        total_distance = np.sum(step_lengths)
        final_displacement = np.linalg.norm(self.position)
        
        print(f"\n=== Route Summary ===")
        print(f"Total distance traveled: {total_distance:.3f} m")
        print(f"Final displacement from start: {final_displacement:.3f} m")
        print(f"Final position: ({self.position[0]:.3f}, {self.position[1]:.3f}) m")
        
        return np.array(self.path), np.array(self.headings)
    
    def visualize_route(self, title="Pedestrian Dead Reckoning - Route Trace"):
        """
        Visualize the traced route.
        
        Args:
            title: plot title
        """
        path = np.array(self.path)
        
        plt.figure(figsize=(14, 12))
        
        # Plot 1: Route trace
        plt.subplot(2, 2, 1)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='End')
        
        # Add step markers every n steps
        step_interval = max(1, len(path) // 15)  # Show about 15 markers
        for i in range(0, len(path), step_interval):
            plt.annotate(f"{i}", (path[i, 0], path[i, 1]), 
                         fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle="circle", fc="white", alpha=0.7))
        
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title('2D Route Trace')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # Plot 2: Heading over time
        plt.subplot(2, 2, 2)
        headings = np.array(self.headings)
        plt.plot(headings, 'r-', linewidth=2)
        plt.xlabel('Step Number')
        plt.ylabel('Heading (degrees)')
        plt.title('Heading Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 360)
        
        # Plot 3: Distance from start
        plt.subplot(2, 2, 3)
        distances = [np.linalg.norm(pos) for pos in path]
        plt.plot(distances, 'g-', linewidth=2)
        plt.xlabel('Step Number')
        plt.ylabel('Distance from Start (m)')
        plt.title('Distance from Starting Point')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Step-by-step displacement
        plt.subplot(2, 2, 4)
        if len(path) > 1:
            step_displacements = [np.linalg.norm(path[i] - path[i-1]) 
                                for i in range(1, len(path))]
            plt.plot(step_displacements, 'm-', linewidth=2)
            plt.xlabel('Step Number')
            plt.ylabel('Step Length (m)')
            plt.title('Individual Step Lengths')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.show()

def retrace_route_from_file(file_path, initial_position=(0, 0), initial_heading=0, K=0.5):
    """
    Complete PDR pipeline to retrace route from sensor data file.
    
    Args:
        file_path: path to sensor data file (without .csv extension)
        initial_position: starting position (x, y) in meters
        initial_heading: initial heading in degrees
        K: Weinberg calibration constant
    
    Returns:
        pdr: PedestrianDeadReckoning instance with traced route
    """
    # Read sensor data
    df = read_file.read_file("../data", file_path)
    
    # Initialize PDR system
    pdr = PedestrianDeadReckoning(initial_position, initial_heading)
    
    # Process data and trace route
    path, headings = pdr.process_sensor_data(df, K=K)
    
    # Visualize results
    pdr.visualize_route(f"PDR Route Trace - {file_path}")
    
    return pdr

if __name__ == "__main__":
    print("=== Pedestrian Dead Reckoning Route Tracing ===")
    print("Optimized for chest-mounted device with screen facing outward")
    
    # Retrace route from sensor data
    file_name = "meow"  # Change this to your actual file name
    
    try:
        pdr = retrace_route_from_file(file_name, initial_position=(0, 0), initial_heading=0, K=0.5)
        pdr.visualize_route(f"PDR Route Trace - {file_name}")
        # Print final statistics
        final_path = np.array(pdr.path)
        print(f"\n=== Final Results ===")
        print(f"Total steps: {len(final_path) - 1}")
        print(f"Final position: ({pdr.position[0]:.3f}, {pdr.position[1]:.3f}) m")
        print(f"Total displacement: {np.linalg.norm(pdr.position):.3f} m")
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Please check that the file exists and contains valid sensor data.")