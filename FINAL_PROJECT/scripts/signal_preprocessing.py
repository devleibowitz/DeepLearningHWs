# script that takes in raw signals and outputs cleaned up signals

import numpy as np
import scipy.interpolate as interp
import copy
import matplotlib as plt
import os
import torch
import wfdb

def preprocess_fhr(fhr, fs=4, interpolation = "cubic"):
    """
    Preprocess fetal heart rate (FHR) data:
    1. Replace extreme values (>210 or <50) and NaNs with 0.
    2. Use cubic spline interpolation for missing (zero) values.

    Parameters:
        fhr (numpy array): The fetal heart rate time series.
        max_missing_duration (int): Maximum duration for interpolation in seconds.

    Returns:
        numpy array: The preprocessed FHR data.
    """
    max_missing_duration=10*60*fs

    # deep copy fhr
    fhr = copy.deepcopy(fhr)

    time = np.arange(len(fhr))

    # Identify missing values (NaNs)
    missing_mask = np.isnan(fhr)

    # Identify extreme values (above 210 or below 50)
    extreme_mask = (fhr > 210) | (fhr < 50)

    # Replace extreme values and NaNs with 0
    fhr[missing_mask | extreme_mask] = 0

    # Identify all zero values for interpolation
    zero_mask = fhr == 0

    # Find valid (nonzero) values
    valid_idx = np.where(fhr != 0)[0]

    if interpolation == "cubic":
        if len(valid_idx) > 3:  # Need at least 4 points for cubic interpolation
            interpolator = interp.CubicSpline(valid_idx, fhr[valid_idx])
            fhr[zero_mask] = interpolator(np.where(zero_mask)[0])
    else:
        if len(valid_idx) > 1:  # Ensure there are at least two points for interpolation
            interpolator = interp.interp1d(valid_idx, fhr[valid_idx], kind='linear', bounds_error=False, fill_value="extrapolate")
            fhr[zero_mask] = interpolator(np.where(zero_mask)[0])

    return fhr

def create_pytorch_tensor(records,process = False):
    """
    Creates a PyTorch tensor from a dictionary of wfdb records.

    Args:
        records: A dictionary where keys are record names and values are wfdb.Record objects.

    Returns:
        A PyTorch tensor of shape (num_records, 2, max_signal_length)
        or None if input is invalid.
    """

    records = dict(sorted(all_records.items()))

    if not isinstance(records, dict):
        print("Error: Input must be a dictionary of wfdb records.")
        return None

    # Find maximum signal length for padding
    max_signal_length = 0
    for record in records.values():
      max_signal_length = max(max_signal_length, len(record.p_signal))

    num_records = len(records)
    tensor_data = []

    for record_name, record in records.items():
        # Pad signals to max length
        fhr = record.p_signal[:,0]
        if process:
          fhr = preprocess_fhr(fhr)
        uc = record.p_signal[:,1]
        fhr_padded = np.pad(fhr, (0, max_signal_length-len(fhr)), 'constant') # padding at the end, switch 0 and max_signal_length-len(fhr)
        uc_padded = np.pad(uc, (0, max_signal_length-len(uc)), 'constant')

        # Stack and append to list
        record_tensor = np.stack([fhr_padded, uc_padded], axis=0)
        tensor_data.append(record_tensor)

    # Convert the list of numpy arrays to a single numpy array
    tensor_data = np.array(tensor_data)
    # Convert to PyTorch tensor
    tensor = torch.tensor(tensor_data, dtype=torch.float32)
    return tensor

def read_records(data_dir):
  """
  Iterates through .dat files in data_dir, reads them using wfdb,
  and stores them in a dictionary.

  Args:
    data_dir: The directory containing the .dat files.

  Returns:
    A dictionary where keys are filenames (without extension) and
    values are wfdb.Record objects.
    Returns an empty dictionary if no .dat files are found or an error occurs.
  """
  records = {}
  for filename in os.listdir(data_dir):
    if filename.endswith(".dat"):
      record_id = filename[:-4]  # Remove the .dat extension
      try:
        record = wfdb.rdrecord(os.path.join(data_dir, record_id))
        records[record_id] = record
      except Exception as e:
        print(f"Error reading {filename}: {e}")
  return records

DATA_DIR = '/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data/'
all_records = read_records(f"{DATA_DIR}/ctu_chb_data")
# Example usage assuming 'all_records' is your dictionary of wfdb records
# Replace this example with your actual records dictionary
#Example usage: Assuming all_records dictionary is available.
pytorch_tensor = create_pytorch_tensor(all_records)
pytorch_tensor_processed = create_pytorch_tensor(all_records, process = True)

if pytorch_tensor is not None:
  print(pytorch_tensor.shape)
  torch.save(pytorch_tensor, f"{DATA_DIR}/processed/ctg_tensor.pt")