import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import detect_steps

def list_csv_files(data_dir):
    """List all CSV files in the data directory"""
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

def select_file(data_dir):
    """Allow user to select a CSV file from the data directory"""
    files = list_csv_files(data_dir)
    
    if not files:
        print("No CSV files found in the data directory.")
        return None
    
    print("Available CSV files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input("Select a file (enter the number): ")) - 1
            if 0 <= choice < len(files):
                return files[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def calculate_chest_mounted_heading(mx, my, mz, ax, ay, az):
    """
    Calculate heading for chest-mounted device (screen facing outward)
    
    Args:
        mx, my, mz: Magnetometer readings
        ax, ay, az: Accelerometer readings
        
    Returns:
        heading: Heading in degrees (0=North, 90=East)
    """
    # Normalize acceleration to estimate gravity direction
    acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if acc_norm < 0.1:
        return 0
    
    gx, gy, gz = ax/acc_norm, ay/acc_norm, az/acc_norm
    
    # Calculate pitch and roll for chest-mounted device
    pitch = np.arctan2(-gy, np.sqrt(gx**2 + gz**2))
    roll = np.arctan2(gx, gz)
    
    # Apply tilt compensation to magnetometer readings
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    
    # Compensate for tilt
    mx_comp = (mx * cos_pitch) + (mz * sin_pitch)
    my_comp = (mx * sin_roll * sin_pitch) + (my * cos_roll) - (mz * sin_roll * cos_pitch)
    mz_comp = (-mx * sin_pitch) + (mz * cos_pitch)  # Added this line to define mz_comp
    
    # Calculate heading
    heading_rad = np.arctan2(mx_comp, mz_comp)
    heading_deg = np.degrees(heading_rad)
    
    # Normalize to 0-360 degrees
    heading_deg = (heading_deg + 360) % 360
    
    return heading_deg

def plot_graphs_for_file(data_dir, filename):
    """Plot graphs for a selected CSV file with chest-mount orientation"""
    file_path = os.path.join(data_dir, filename)
    
    df = pd.read_csv(file_path)
    
    required_cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns in {filename}: {missing_cols}")
        return
    
    output_dir = os.path.join(data_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate SVM
    df['svm'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    # Apply low-pass filter to reduce noise
    filtered_svm = detect_steps.apply_lowpass_filter(df['svm'].values, cutoff_freq=15.0, sampling_rate=50.0)
    df['svm_filtered'] = filtered_svm
    
    # Calculate chest-mounted heading
    headings = []
    for i in range(len(df)):
        heading = calculate_chest_mounted_heading(
            df['mx'].iloc[i], df['my'].iloc[i], df['mz'].iloc[i],
            df['ax'].iloc[i], df['ay'].iloc[i], df['az'].iloc[i]
        )
        headings.append(heading)
    df['chest_heading'] = headings
    
    # Apply moving average to smooth headings
    df['chest_heading_smooth'] = df['chest_heading'].rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    
    # Detect steps using threshold method on filtered data
    threshold = np.mean(filtered_svm) + 0.5 * np.std(filtered_svm)  # Adaptive threshold
    step_count, step_times, step_indices = detect_steps.detect_steps_threshold(
        filtered_svm, df['timestamp'].values, threshold, min_time_between_steps=0.3
    )
    
    print(f"Detected {step_count} steps in {filename}")
    print(f"Using threshold: {threshold:.3f}")
    print(f"Filter applied: 15.0 Hz low-pass filter (gentler)")
    
    # Create multi-plot figure
    plt.figure(figsize=(15, 12))
    
    # Create subplot layout
    plt.subplot(3, 2, 1)
    plt.plot(df['timestamp'], df['ax'], label='X-axis', color='red', alpha=0.7)
    plt.plot(df['timestamp'], df['ay'], label='Y-axis', color='green', alpha=0.7)
    plt.plot(df['timestamp'], df['az'], label='Z-axis', color='blue', alpha=0.7)
    
    # Add step markers on accelerometer plot
    if len(step_times) > 0:
        for step_time in step_times:
            plt.axvline(x=step_time, color='black', linestyle='--', alpha=0.8, linewidth=1)
        plt.plot([], [], color='black', linestyle='--', alpha=0.8, label=f'Steps ({step_count})')
    
    plt.title(f'Accelerometer Data with Step Detection - {filename}')
    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration (g)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(df['timestamp'], df['svm'], label='Original SVM', color='gray', linewidth=1, alpha=0.5)
    plt.plot(df['timestamp'], df['svm_filtered'], label='Filtered SVM', color='purple', linewidth=2)
    
    # Add threshold line and step markers on SVM plot
    plt.axhline(y=threshold, color='orange', linestyle='-', alpha=0.8, linewidth=2, label=f'Threshold ({threshold:.2f})')
    if len(step_times) > 0:
        for step_time in step_times:
            plt.axvline(x=step_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        plt.plot([], [], color='red', linestyle='--', alpha=0.8, label=f'Steps ({step_count})')
    
    plt.title(f'Filtered SVM with Step Detection - {filename}')
    plt.xlabel('Timestamp')
    plt.ylabel('SVM (g)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a subplot showing magnetometer data
    plt.subplot(3, 2, 3)
    plt.plot(df['timestamp'], df['mx'], label='X-axis', color='red', alpha=0.7)
    plt.plot(df['timestamp'], df['my'], label='Y-axis', color='green', alpha=0.7)
    plt.plot(df['timestamp'], df['mz'], label='Z-axis', color='blue', alpha=0.7)
    
    plt.title(f'Magnetometer Data - {filename}')
    plt.xlabel('Timestamp')
    plt.ylabel('Magnetic Field (μT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a subplot showing gyroscope data
    plt.subplot(3, 2, 4)
    plt.plot(df['timestamp'], df['gx'], label='X-axis', color='red', alpha=0.7)
    plt.plot(df['timestamp'], df['gy'], label='Y-axis', color='green', alpha=0.7)
    plt.plot(df['timestamp'], df['gz'], label='Z-axis', color='blue', alpha=0.7)
    
    plt.title(f'Gyroscope Data - {filename}')
    plt.xlabel('Timestamp')
    plt.ylabel('Angular Velocity (°/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a subplot showing the calculated chest-mounted heading
    plt.subplot(3, 2, 5)
    plt.plot(df['timestamp'], df['chest_heading'], label='Raw Heading', color='blue', alpha=0.5)
    plt.plot(df['timestamp'], df['chest_heading_smooth'], label='Smoothed Heading', color='red', linewidth=2)
    
    if len(step_times) > 0:
        step_headings = [df.loc[df['timestamp'] == st, 'chest_heading_smooth'].iloc[0] if len(df.loc[df['timestamp'] == st]) > 0 else 0 for st in step_times]
        plt.scatter(step_times, step_headings, color='green', s=50, label='Step Headings')
    
    plt.title(f'Chest-Mounted Heading - {filename}')
    plt.xlabel('Timestamp')
    plt.ylabel('Heading (degrees)')
    plt.ylim(0, 360)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add step statistics subplot
    plt.subplot(3, 2, 6)
    if len(step_times) > 1:
        step_intervals = np.diff(step_times)
        plt.hist(step_intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Step Interval Distribution - {filename}')
        plt.xlabel('Time between steps')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_interval = np.mean(step_intervals)
        plt.text(0.02, 0.98, f'Total Steps: {step_count}\nAvg Interval: {avg_interval:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, f'Steps detected: {step_count}\n(Need >1 step for interval analysis)', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.title(f'Step Statistics - {filename}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'analysis_{filename.replace(".csv", "")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a specialized plot for chest-mounted orientation
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Chest-Mounted Device Analysis - {filename}', fontsize=16)
    
    # Show device orientation diagram
    plt.subplot(2, 2, 1)
    plt.text(0.5, 0.5, 'Device Orientation:\n\nScreen facing outward\nChest-mounted\n\nZ-axis: Forward\nY-axis: Down\nX-axis: Right', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.axis('off')
    
    # Show magnetometer pattern in XZ plane (horizontal)
    plt.subplot(2, 2, 2)
    plt.scatter(df['mx'], df['mz'], c=df['timestamp'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Time')
    plt.title('Magnetometer XZ-Plane (Horizontal)')
    plt.xlabel('mx')
    plt.ylabel('mz')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Show accelerometer during steps
    plt.subplot(2, 2, 3)
    if step_count > 0:
        # Get windows around steps
        window = 10  # points before/after step
        for i, idx in enumerate(step_indices[:min(5, len(step_indices))]):  # Show first 5 steps
            start_idx = max(0, idx - window)
            end_idx = min(len(df), idx + window)
            plt.plot(df['timestamp'].iloc[start_idx:end_idx], 
                     df['svm_filtered'].iloc[start_idx:end_idx], 
                     label=f'Step {i+1}')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        plt.title('Acceleration Patterns During Steps')
        plt.xlabel('Timestamp')
        plt.ylabel('SVM (g)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No steps detected', ha='center', va='center')
    plt.grid(True, alpha=0.3)
    
    # Show heading rose for steps
    plt.subplot(2, 2, 4, polar=True)
    if step_count > 0:
        step_headings = [df.loc[df['timestamp'] == st, 'chest_heading_smooth'].iloc[0] if len(df.loc[df['timestamp'] == st]) > 0 else 0 for st in step_times]
        angles = np.radians(step_headings)
        # Create histogram of heading angles
        bins = np.linspace(0, 2*np.pi, 16)
        hist, _ = np.histogram(angles, bins)
        width = 2*np.pi / 15
        plt.bar(bins[:-1], hist, width=width, alpha=0.7)
        plt.title('Step Direction Distribution')
    else:
        plt.text(0, 0, 'No steps detected', ha='center', va='center')
    
    # Add N, E, S, W labels to polar plot
    plt.text(0, 1.1, 'N', transform=plt.gca().transAxes, ha='center')
    plt.text(1.1, 0.5, 'E', transform=plt.gca().transAxes, va='center')
    plt.text(0.5, -0.1, 'S', transform=plt.gca().transAxes, ha='center')
    plt.text(-0.1, 0.5, 'W', transform=plt.gca().transAxes, va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'chest_mount_analysis_{filename.replace(".csv", "")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved for {filename} in {output_dir}")

def main():
    """Main function to run the plotting program"""
    data_dir = '../data' 
    
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    selected_file = select_file(data_dir)
    if selected_file:
        print(f"Selected file: {selected_file}")
        plot_graphs_for_file(data_dir, selected_file)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()