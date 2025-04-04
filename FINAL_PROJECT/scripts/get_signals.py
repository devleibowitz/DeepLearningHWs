import os
import numpy as np
import torch
from scipy import interpolate
from scipy.io import loadmat
import json
import wfdb
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

DATA_DIR = "/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data"

def trim_zeros(signal, threshold=5):
    """Remove only leading and trailing segments where all values are below a given threshold."""
    
    # Validation checks
    if not isinstance(signal, np.ndarray):
        raise TypeError("The input 'signal' must be a numpy array.")
    
    if not isinstance(threshold, (int, float)):
        raise ValueError("The 'threshold' must be a numerical value (int or float).")
    
    if len(signal) == 0:
        raise ValueError("The input 'signal' cannot be an empty array.")
    
    above_threshold_indices = np.where(signal > threshold)[0]

    if len(above_threshold_indices) == 0:
        print("No values below threshold. Returning the original signal.")
        return signal
    
    first_valid_index = above_threshold_indices[0]
    last_valid_index = above_threshold_indices[-1]

    trimmed_signal = signal[first_valid_index:last_valid_index + 1]
    return trimmed_signal

def preprocess_signal(signal, threshold=5, UC=False):
    """Preprocess a fetal heart rate signal by removing zero segments, interpolating invalid points, 
    normalizing, and trimming trailing zeros."""
    # Step 1: Remove leading and trailing zero segments
    signal = trim_zeros(signal, threshold)
    
    if len(signal) == 0:
        print("Skipping empty signal")
        return np.array([])  # Skip empty signals

    # Step 2: Identify invalid points (0 bpm or out of range 50-210 bpm)
    invalid_mask = (signal == 0) | (signal < 50) | (signal > 210)
    if UC:
        invalid_mask = (signal > 120)
    
    if np.all(invalid_mask):
        print("Discarding signal")
        return np.array([])  # If all values are invalid, discard signal

    # Step 3: Interpolate missing/invalid values using cubic spline
    valid_indices = np.where(~invalid_mask)[0]
    valid_values = signal[valid_indices]
    
    interpolator = CubicSpline(valid_indices, valid_values, bc_type='not-a-knot', extrapolate=True)
    signal = interpolator(np.arange(len(signal)))

    # Step 4: Normalize the signal (min-max scaling)
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val > min_val:
        signal = (signal - min_val) / (max_val - min_val)
        
    else:
        signal = np.zeros_like(signal)  # If all values are the same, return zero array

    # Step 5: Remove any remaining trailing zeros after normalization
    # signal = trim_zeros(signal, threshold=0)
    return signal

# process both FHR and UC
def process_ctu_chb_signals(data_dir):
    signals_list = []
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .dat files and sort by ID
    dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".dat")], key=lambda x: int(x.split(".")[0]))
    
    for dat_file in dat_files:
        file_id = dat_file.split('.')[0]  # Extract ID from filename
        record_path = os.path.join(data_dir, file_id)
        
        try:
            # Read the .dat file using WFDB
            record = wfdb.rdsamp(record_path)  # Returns (signals, fields)
            signals = record[0]  # shape: [num_samples, num_channels]
            
            # Extract FHR and UC signals (assuming FHR is channel 0 and UC is channel 1)
            fhr = signals[:, 0]  # Fetal Heart Rate
            uc = signals[:, 1]   # Uterine Contractions

            
            # Preprocess both FHR and UC signals
            processed_fhr = preprocess_signal(fhr)
            processed_uc = preprocess_signal(uc, threshold=0, UC=True)

            if processed_fhr is not None and processed_uc is not None:
                # Find the maximum length of the two signals
                max_len = max(len(processed_fhr), len(processed_uc))
                
                # Pad shorter signal with zeros
                padded_fhr = torch.tensor(processed_fhr, dtype=torch.float32)
                padded_uc = torch.tensor(processed_uc, dtype=torch.float32)
                
                # Padding shorter signal with zeros
                if len(padded_fhr) < max_len:
                    padded_fhr = torch.cat([padded_fhr, torch.zeros(max_len - len(padded_fhr))])
                if len(padded_uc) < max_len:
                    padded_uc = torch.cat([padded_uc, torch.zeros(max_len - len(padded_uc))])
                
                # Stack both signals into a single tensor (shape: [max_len, 2])
                combined_signal = torch.stack([padded_fhr, padded_uc], dim=-1)  # shape: [num_samples, 2]
                signals_list.append(combined_signal)

        except Exception as e:
            print(f"Skipping {dat_file} due to error: {e}")
            continue
    
    if len(signals_list) == 0:
        print("No valid signals found!")
        return None
    
    # Pad signals to the longest sequence length
    max_length = max(sig.shape[0] for sig in signals_list)
    padded_signals = torch.zeros(len(signals_list), max_length, 2)  # 2 for FHR and UC channels
    
    for i, sig in enumerate(signals_list):
        padded_signals[i, :sig.shape[0], :] = sig  # Left-align and zero-pad shorter signals
    
    # Save tensor
    output_path = os.path.join(output_dir, "fhr_uc_signal_tensor.pt")
    torch.save(padded_signals, output_path)
    print(f"Processed signal tensor saved at {output_path}")

    # Save mapping
    id_mapping_path = os.path.join(output_dir, "signal_id_mapping.json")
    id_mapping = {int(dat_file.split(".")[0]): idx for idx, dat_file in enumerate(dat_files)}

    with open(id_mapping_path, "w") as f:
        json.dump(id_mapping, f)

    print(f"Signal ID mapping saved at {id_mapping_path}")
    
    return padded_signals

def get_signal_by_id(data_dir, signal_id):
    tensor_path = os.path.join(data_dir, "processed", "signal_tensor.pt")
    mapping_path = os.path.join(data_dir, "processed", "signal_id_mapping.json")

    # Load tensor and mapping
    signals = torch.load(tensor_path)
    with open(mapping_path, "r") as f:
        id_mapping = json.load(f)

    # Get the tensor index for the given signal ID
    tensor_index = id_mapping.get(str(signal_id))  # Ensure ID is a string for lookup

    if tensor_index is None:
        raise ValueError(f"Signal ID {signal_id} not found in dataset.")

    return signals[tensor_index]

def plot_signal_comparison(data_dir, signal_id, UC=False, tensor_file="signal_tensor.pt"):
    """Plots the raw vs. processed signal for a given ID. If UC=True, plots both FHR and UC."""
    raw_data_path = os.path.join(data_dir, "ctu_chb_data", str(signal_id))
    tensor_path = os.path.join(data_dir, "processed", tensor_file)
    mapping_path = os.path.join(data_dir, "processed", "signal_id_mapping.json")

    # Load raw signal using WFDB
    try:
        record = wfdb.rdsamp(raw_data_path)  # (signals, fields)
        raw_signals = record[0]  # shape: [num_samples, num_channels]
        raw_fhr = raw_signals[:, 0]  # FHR (First channel)
        raw_uc = raw_signals[:, 1]   # UC (Second channel)
    except Exception as e:
        print(f"Error reading raw signal {signal_id}: {e}")
        return
    
    # Load processed tensor and mapping
    try:
        processed_signals = torch.load(tensor_path)
        with open(mapping_path, "r") as f:
            id_mapping = json.load(f)
        
        # Get tensor index for signal ID
        tensor_index = id_mapping.get(str(signal_id))
        if tensor_index is None:
            raise ValueError(f"Signal ID {signal_id} not found in processed dataset.")
        
        processed_signal = processed_signals[tensor_index].numpy()  # [num_samples, 2]
        processed_fhr = processed_signal[:, 0]  # Processed FHR
        processed_uc = processed_signal[:, 1]   # Processed UC
        processed_fhr = trim_zeros(processed_fhr, threshold=0)
        processed_uc = trim_zeros(processed_uc, threshold=0)
        
        print(f'Visualized Processed signal len: {len(processed_fhr)}')
    except Exception as e:
        print(f"Error reading processed signal {signal_id}: {e}")
        return

    # Plot the raw vs. processed signal
    if UC:
        # Plot both FHR and UC (Raw vs. Processed)
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Plot raw FHR and UC
        axs[0].plot(raw_fhr, color='red', label='Raw FHR')
        axs[0].set_title(f'Raw FHR (ID: {signal_id})')
        axs[0].set_ylabel('FHR (bpm)')
        axs[0].legend()
        
        axs[2].plot(raw_uc, color='blue', label='Raw UC')
        axs[2].set_title(f'Raw UC (ID: {signal_id})')
        axs[2].set_ylabel('UC')
        axs[2].legend()

        # Plot processed FHR and UC
        axs[1].plot(processed_fhr, color='green', label='Processed FHR')
        axs[1].set_title(f'Processed FHR (ID: {signal_id})')
        axs[1].set_ylabel('Normalized Value')
        axs[1].legend()

        axs[3].plot(processed_uc, color='purple', label='Processed UC')
        axs[3].set_title(f'Processed UC (ID: {signal_id})')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Normalized Value')
        axs[3].legend()

    else:
        # Plot just FHR (Raw vs. Processed)
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axs[0].plot(raw_fhr, color='red', label='Raw FHR')
        axs[0].set_title(f'Raw FHR (ID: {signal_id})')
        axs[0].set_ylabel('FHR (bpm)')
        axs[0].legend()
        
        axs[1].plot(processed_fhr, color='green', label='Processed FHR')
        axs[1].set_title(f'Processed FHR (ID: {signal_id})')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Normalized Value')
        axs[1].legend()

    plt.tight_layout()
    plt.show()

def get_signal_stats(tensor_path, sampling_rate=4):
    """
    Compute statistics on processed signals.

    Args:
        tensor (np.ndarray): A NumPy array of shape (num_samples, signal_length),
                             where each row represents a processed signal.
        sampling_rate (float): The sampling rate of the signals in Hz (samples per second).

    Returns:
        dict: A dictionary containing min, max, and average signal duration in minutes.
    """

    tensor = torch.load(tensor_path)
    tensor = tensor.cpu().numpy() 
    if not isinstance(tensor, np.ndarray):
        raise TypeError("The input tensor must be a NumPy array.")

    if len(tensor.shape) != 2:
        raise ValueError("Expected a 2D tensor of shape (num_samples, signal_length).")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive number.")

    # Compute signal durations in minutes
    signal_lengths = np.sum(tensor != 0, axis=1)  # Count non-zero values per row
    durations = signal_lengths / sampling_rate / 60  # Convert samples to minutes

    # Compute statistics
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    avg_duration = np.mean(durations)

    stats = {
        "min_duration_min": min_duration,
        "max_duration_min": max_duration,
        "avg_duration_min": avg_duration,
        "num_signals": tensor.shape[0]
    }
    
    return stats


# Example usage:
process_ctu_chb_signals(f'{DATA_DIR}/ctu_chb_data')
# signal = get_signal_by_id(f'{DATA_DIR}', 2004)  # Get signal for ID 12
# plot_signal_comparison(f'{DATA_DIR}', 2005)

tensor_path = os.path.join(DATA_DIR, "processed", "signal_tensor.pt")
tensor_path = os.path.join(DATA_DIR, "processed", "fhr_uc_signal_tensor.pt")
# stats = get_signal_stats(tensor_path)
# print(stats)

# plot_signal_comparison(DATA_DIR, 2004, UC=True, tensor_file="fhr_uc_signal_tensor.pt")
