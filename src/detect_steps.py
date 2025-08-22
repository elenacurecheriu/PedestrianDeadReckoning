## step detection using thresholds

import numpy as np
from scipy.signal import butter, filtfilt
import read_file

def apply_lowpass_filter(data, cutoff_freq=15.0, sampling_rate=50.0, order=2):
    """
    Apply a low-pass Butterworth filter to reduce noise
    
    Args:
        data: array of sensor data
        cutoff_freq: cutoff frequency in Hz (default: 15.0 Hz - less aggressive)
        sampling_rate: sampling rate in Hz (default: 50.0 Hz)
        order: filter order (default: 2 - gentler filtering)
    
    Returns:
        filtered_data: filtered sensor data
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def detect_steps_threshold(svm_data, timestamps, threshold, min_time_between_steps=0.25):
    """
    Detect steps using threshold crossing method with minimum time constraint
    
    Args:
        svm_data: array of SVM values
        timestamps: array of timestamp values
        threshold: threshold value for step detection
        min_time_between_steps: minimum time (seconds) between steps to avoid false positives
    
    Returns:
        step_count: number of steps detected
        step_times: timestamps of detected steps
        step_indices: indices of detected steps
    """
    svm_arr = np.asarray(svm_data, dtype=float)
    ts_arr = np.asarray(timestamps, dtype=float)

    if len(ts_arr) > 1:
        median_dt = np.median(np.diff(ts_arr))
        if median_dt > 1.0:  # likely milliseconds
            ts_arr = ts_arr / 1000.0

    valid_mask = np.isfinite(svm_arr) & np.isfinite(ts_arr)
    valid_idx = np.nonzero(valid_mask)[0]
    svm_valid = svm_arr[valid_mask]
    ts_valid = ts_arr[valid_mask]

    last_state = 'below'
    crossing_count = 0
    crossing_timestamps = []
    crossing_indices = []
    last_step_time = -np.inf

    for j in range(len(svm_valid)):
        svm = svm_valid[j]
        current_time = ts_valid[j]

        if svm > threshold and last_state == 'below':
            if (current_time - last_step_time) >= min_time_between_steps or crossing_count == 0:
                crossing_count += 1
                crossing_timestamps.append(current_time)
                crossing_indices.append(int(valid_idx[j]))
                last_step_time = current_time
            last_state = 'above'
        elif svm < threshold and last_state == 'above':
            last_state = 'below'

    return crossing_count, crossing_timestamps, crossing_indices


if __name__ == "__main__":
    df = read_file.read_file("../data", "log_2")
    
    svm_data = df['svm'].values.astype(float)
    timestamps = df['timestamp'].values.astype(float)

    filtered_svm = apply_lowpass_filter(svm_data, cutoff_freq=15.0, sampling_rate=50.0)
    
    print(f"Original SVM std: {np.std(svm_data):.3f}") #standard deviation - how spread out are the values around the median value
    print(f"Filtered SVM std: {np.std(filtered_svm):.3f}")
    print(f"Noise reduction: {((np.std(svm_data) - np.std(filtered_svm)) / np.std(svm_data) * 100):.1f}%")
    
    adaptive_threshold = np.mean(filtered_svm) + 0.5 * np.std(filtered_svm)
    
    step_count_threshold, step_times_threshold, crossing_indices = detect_steps_threshold(
        filtered_svm, timestamps, adaptive_threshold
    )
    
    print(f"Steps detected: {step_count_threshold}")
    
    
    print("\nComparison with unfiltered data:")
    unfiltered_threshold = np.mean(svm_data) + 0.5 * np.std(svm_data)
    step_count_unfiltered, _, _ = detect_steps_threshold(svm_data, timestamps, unfiltered_threshold)
    
    print(f"Unfiltered steps: {step_count_unfiltered}")
    print(f"Filtered steps: {step_count_threshold}")
    print(f"Difference: {step_count_threshold - step_count_unfiltered}")
    