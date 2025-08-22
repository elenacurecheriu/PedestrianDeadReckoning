import numpy as np
import read_file
from detect_steps import detect_steps_threshold, apply_lowpass_filter

def extract_step_accelerations(svm_data, step_indices, window_size=10):
    """
    Extract acceleration values around each detected step for step length estimation.
    
    Args:
        svm_data: array of SVM values
        step_indices: array of step detection indices
        window_size: window size around each step to extract min/max/avg values
    
    Returns:
        step_accelerations: dictionary containing acceleration statistics for each step
    """
    step_accelerations = {
        'a_max': [],
        'a_min': [],
        'a_avg': [],
        'a_range': []
    }
    
    for step_idx in step_indices:
        start_idx = max(0, step_idx - window_size)
        end_idx = min(len(svm_data), step_idx + window_size)
        
        window_data = svm_data[start_idx:end_idx]
        
        a_max = np.max(window_data)
        a_min = np.min(window_data)
        a_avg = np.mean(window_data)
        a_range = a_max - a_min
        
        step_accelerations['a_max'].append(a_max)
        step_accelerations['a_min'].append(a_min)
        step_accelerations['a_avg'].append(a_avg)
        step_accelerations['a_range'].append(a_range)
    
    return step_accelerations

def weinberg_step_length(a_max, a_min, K=0.5):
    """
    Weinberg's method for step length estimation.
    
    Formula: step_length = K × (a_max - a_min)^(1/4)
    
    Args:
        a_max: maximum acceleration in the step window
        a_min: minimum acceleration in the step window
        K: calibration constant (default: 0.5)
    
    Returns:
        step_length: estimated step length in meters
    """
    a_range = np.array(a_max) - np.array(a_min)
    step_length = K * np.power(a_range, 1/4)
    return step_length

def estimate_step_lengths(df, step_indices, K=0.5, window_size=10):
    """
    Estimate step lengths using Weinberg's method.

    Args:
        df: DataFrame with accelerometer data
        step_indices: indices of detected steps
        K: calibration constant
        window_size: window size around each step

    Returns:
        step_lengths: array of estimated step lengths
        step_accelerations: dictionary of acceleration statistics
    """
    svm_data = df['svm'].values
    step_accelerations = extract_step_accelerations(svm_data, step_indices, window_size)
    step_lengths = weinberg_step_length(
        step_accelerations['a_max'],
        step_accelerations['a_min'],
        K
    )

    return step_lengths, step_accelerations


if __name__ == "__main__":
    df = read_file.read_file("../data", "meow")
    
    svm_data = df['svm'].values
    timestamps = df['timestamp'].values
    
    filtered_svm = apply_lowpass_filter(svm_data, cutoff_freq=15.0, sampling_rate=50.0)
    
    adaptive_threshold = np.mean(filtered_svm) + 0.5 * np.std(filtered_svm)
    step_count, step_times, step_indices = detect_steps_threshold(
        filtered_svm, timestamps, adaptive_threshold
    )
    
    print(f"Steps detected: {step_count}")
    
    if step_count > 0:
        step_lengths_weinberg, step_accelerations = estimate_step_lengths(
            df, step_indices, K=0.5, window_size=10
        )
        total_distance = np.sum(step_lengths_weinberg)
        print(f"Total distance: {total_distance:.3f} m")
        
        def get_total_distance():
            return total_distance
            
    else:
        print("No steps detected")
        def get_total_distance():
            return 0.0