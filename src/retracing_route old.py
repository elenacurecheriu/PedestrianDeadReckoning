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
        mz_comp = (-mx_cal * sin_pitch) + (mz_cal * cos_pitch)  # Define mz_comp
        
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

    def compute_gyro_heading(self, timestamps, gx, gy, gz, initial_heading=None):
        """
        Compute heading based on gyroscope integration.
        For chest-mounted device (screen facing outward), the heading change
        is primarily from rotation around the y-axis.
        
        Args:
            timestamps: Array of timestamp values
            gx, gy, gz: Gyroscope readings (degrees/s)
            initial_heading: Optional initial heading (degrees)
            
        Returns:
            gyro_headings: Array of headings computed from gyroscope
        """
        # For chest-mounted orientation (screen outward):
        # - When standing still, Y-axis points down
        # - Z-axis points forward (away from chest)
        # - X-axis points right
        # 
        # When turning right (clockwise):
        # - Primary rotation is around Y-axis
        # - Negative gy value indicates clockwise rotation (turning right)
        # - Positive gy value indicates counter-clockwise rotation (turning left)
        
        # Initialize heading array
        gyro_headings = np.zeros_like(timestamps, dtype=float)
        
        # Set initial heading
        if initial_heading is not None:
            gyro_headings[0] = initial_heading
        
        # Integrate gyroscope data to get heading
        for i in range(1, len(timestamps)):
            # Time delta in seconds
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # Assuming timestamps in ms
            
            # For chest-mounted device, we need to consider the device orientation
            # The heading change comes primarily from the Y-axis gyroscope for this orientation
            # We need to negate the value because positive gy means counter-clockwise rotation
            # but increasing heading means clockwise (e.g., North to East)
            heading_change = -gy[i] * dt
            
            # Update heading
            gyro_headings[i] = gyro_headings[i-1] + heading_change
            
            # Normalize to 0-360 degrees
            gyro_headings[i] = gyro_headings[i] % 360
        
        return gyro_headings

    def fuse_headings(self, mag_headings, gyro_headings, alpha=0.98):
        """
        Fuse magnetometer and gyroscope headings using a complementary filter.
        
        Args:
            mag_headings: Array of magnetometer-based headings
            gyro_headings: Array of gyroscope-based headings
            alpha: Weight for gyroscope data (0-1), higher values favor gyro
            
        Returns:
            fused_headings: Array of fused headings
        """
        # Initialize fused heading array
        fused_headings = np.zeros_like(mag_headings)
        fused_headings[0] = mag_headings[0]  # Start with magnetometer heading
        
        for i in range(1, len(mag_headings)):
            # Handle heading wrap-around (e.g., 359° to 0°)
            if abs(mag_headings[i] - gyro_headings[i]) > 180:
                if mag_headings[i] > gyro_headings[i]:
                    adjusted_mag = mag_headings[i] - 360
                else:
                    adjusted_mag = mag_headings[i] + 360
            else:
                adjusted_mag = mag_headings[i]
                
            # Apply complementary filter
            # Use more gyro (alpha) for short-term changes, but rely on mag for long-term stability
            fused_headings[i] = alpha * gyro_headings[i] + (1-alpha) * adjusted_mag
            
            # Normalize to 0-360 degrees
            fused_headings[i] = fused_headings[i] % 360
            
        return fused_headings
    
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
    
    def process_sensor_data(self, df, K=0.5, window_size=10, gyro_weight=0.98):
        """
        Process complete sensor data to retrace route.
        
        Args:
            df: DataFrame with sensor data
            K: Weinberg calibration constant
            window_size: window size for step detection
            gyro_weight: Weight for gyroscope data in heading fusion (0-1)
        
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
        gx = df['gx'].values.astype(float)
        gy = df['gy'].values.astype(float)
        gz = df['gz'].values.astype(float)
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
        
        # Calculate magnetometer-based headings
        print("=== Computing Headings ===")
        mag_headings = []
        for i in range(len(df)):
            _, _, _, heading = self.chest_mounted_to_earth_frame(
                ax[i], ay[i], az[i], mx[i], my[i], mz[i]
            )
            mag_headings.append(heading)
        
        # Calculate gyroscope-based headings
        gyro_headings = self.compute_gyro_heading(
            timestamps, gx, gy, gz, initial_heading=mag_headings[0]
        )
        
        # Fuse magnetometer and gyroscope headings
        fused_headings = self.fuse_headings(
            mag_headings, gyro_headings, alpha=gyro_weight
        )
        
        # Smooth the fused headings
        if len(fused_headings) >= 5:
            smoothed_headings = moving_average(fused_headings, 5)
            # Pad the beginning to match original length
            pad_length = len(fused_headings) - len(smoothed_headings)
            smoothed_headings = np.pad(smoothed_headings, (pad_length, 0), 'edge')
        else:
            smoothed_headings = fused_headings
        
        print(f"Average step length: {np.mean(step_lengths):.3f} m")
        print(f"Using gyro-magnetometer fusion with alpha={gyro_weight}")
        
        # Process each step
        print("=== Tracing Route with Gyro-Enhanced Heading ===")
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
        
        # Store headings data for visualization
        self.mag_headings = mag_headings
        self.gyro_headings = gyro_headings
        self.fused_headings = fused_headings
        self.smoothed_headings = smoothed_headings
        
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
        
        # Plot 2: Heading comparison
        plt.subplot(2, 2, 2)
        if hasattr(self, 'mag_headings') and hasattr(self, 'gyro_headings') and hasattr(self, 'fused_headings'):
            # Create array for x-axis (indices)
            indices = np.arange(len(self.mag_headings))
            
            # Plot sample of headings for clarity
            sample_rate = max(1, len(indices) // 500)  # Sample to avoid overcrowding
            plt.plot(indices[::sample_rate], np.array(self.mag_headings)[::sample_rate], 'g-', alpha=0.5, label='Magnetometer')
            plt.plot(indices[::sample_rate], np.array(self.gyro_headings)[::sample_rate], 'r-', alpha=0.5, label='Gyroscope')
            plt.plot(indices[::sample_rate], np.array(self.fused_headings)[::sample_rate], 'b-', linewidth=2, label='Fused')
            plt.xlabel('Sample Index')
            plt.ylabel('Heading (degrees)')
            plt.title('Heading Comparison')
            plt.legend()
        else:
            # If no heading data is available, just show the headings used for path
            headings = np.array(self.headings)
            plt.plot(headings, 'r-', linewidth=2, label='Heading')
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

def retrace_route_from_file(file_path, initial_position=(0, 0), initial_heading=0, K=0.5, gyro_weight=0.98):
    """
    Complete PDR pipeline to retrace route from sensor data file.
    
    Args:
        file_path: path to sensor data file (without .csv extension)
        initial_position: starting position (x, y) in meters
        initial_heading: initial heading in degrees
        K: Weinberg calibration constant
        gyro_weight: Weight for gyroscope data in heading fusion (0-1)
    
    Returns:
        pdr: PedestrianDeadReckoning instance with traced route
    """
    # Read sensor data
    df = read_file.read_file("../data", file_path)
    
    # Initialize PDR system
    pdr = PedestrianDeadReckoning(initial_position, initial_heading)
    
    # Process data and trace route
    path, headings = pdr.process_sensor_data(df, K=K, gyro_weight=gyro_weight)
    
    # Visualize results
    pdr.visualize_route(f"PDR Route Trace - {file_path}")
    
    return pdr

if __name__ == "__main__":
    print("=== Pedestrian Dead Reckoning Route Tracing ===")
    print("Optimized for chest-mounted device with gyroscope integration")
    
    # Retrace route from sensor data
    file_name = "log_6"  # Change this to your actual file name

    try:
        # You can adjust the gyro_weight parameter to tune the fusion
        # Higher values (closer to 1) give more weight to the gyroscope data
        pdr = retrace_route_from_file(file_name, initial_position=(0, 0), initial_heading=0, K=0.3, gyro_weight=0.98)
        
        # Print final statistics
        final_path = np.array(pdr.path)
        print(f"\n=== Final Results ===")
        print(f"Total steps: {len(final_path) - 1}")
        print(f"Final position: ({pdr.position[0]:.3f}, {pdr.position[1]:.3f}) m")
        print(f"Total displacement: {np.linalg.norm(pdr.position):.3f} m")
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Please check that the file exists and contains valid sensor data.")