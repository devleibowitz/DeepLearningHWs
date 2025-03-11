# script that takes in raw signals and outputs cleaned up signals

import numpy as np
import scipy.interpolate as interp
import copy
import torch
import pandas as pd
import wfdb
import os


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


def preprocess_time_series(time_series, extrapolation=True, above_threshold=210, below_threshold=50, record_name="",
                           print=False):
    """
    Preprocess time_series such as fetal heart rate (FHR) data:
    1. Replace extreme values (>210 or <50) and NaNs with 0.
    2. Use cubic spline interpolation for missing (zero) values.

    Parameters:
        below_threshold:
        above_threshold:
        extrapolation:
        time_series:
        fhr (numpy array): The fetal heart rate time series.
        max_missing_duration (int): Maximum duration for interpolation in seconds.

    Returns:
        numpy array: The preprocessed FHR data.
        True if no valid data points to extrapolate, False otherwise.
    """

    # deep copy fhr
    time_series = copy.deepcopy(time_series)

    time = np.arange(len(time_series))

    # Identify missing values (NaNs)
    missing_mask = np.isnan(time_series)

    # Identify extreme values (above 210 or below 50)
    extreme_mask = (time_series > above_threshold) | (time_series < below_threshold)

    # Replace extreme values and NaNs with 0
    time_series[missing_mask | extreme_mask] = 0
    # Replace extreme values and NaNs with 0
    time_series[missing_mask | extreme_mask] = 0

    # Identify all zero values for interpolation
    zero_mask = time_series == 0

    # Find valid (nonzero) values
    valid_idx = np.where(time_series != 0)[0]

    if len(valid_idx) < 2:
        if print == True: print(
            f"Warning: {record_name}, Not enough valid data points for interpolation. Returning original signal.")
        return time_series, True

    interpolator = interp.CubicSpline(valid_idx, time_series[valid_idx], extrapolate=extrapolation)
    time_series[zero_mask] = interpolator(np.where(zero_mask)[0])

    return time_series, False


def create_pytorch_tensor(records, process=False):
    """
    Creates a PyTorch tensor from a dictionary of wfdb records.

    Args:
        process:
        records: A dictionary where keys are record names and values are wfdb.Record objects.

    Returns:
        tensor: A PyTorch tensor of shape (num_records, 2, max_signal_length)
        or None if input is invalid.

        invalid_records: invalid to interpolate records

    """

    records = dict(sorted(all_records.items()))  # Labels are sorted by name.

    if not isinstance(records, dict):
        print("Error: Input must be a dictionary of wfdb records.")
        return None, None

    # Find maximum signal length for padding
    max_signal_length = 0
    for record in records.values():
        max_signal_length = max(max_signal_length, len(record.p_signal))

    num_records = len(records)
    tensor_data = []
    invalid_records = []
    flag_fhr = flag_uc = False

    for record_name, record in records.items():
        # Pad signals to max length
        fhr = record.p_signal[:, 0]
        uc = record.p_signal[:, 1]

        if len(fhr) != len(uc):
            print("len(fhr)!=len(uc)")

        if process:
            fhr, flag_fhr = preprocess_time_series(fhr, extrapolation=False)
            uc, flag_uc = preprocess_time_series(uc, extrapolation=False, above_threshold=100, below_threshold=4,
                                                 record_name=record_name)

        if flag_fhr | flag_uc:
            invalid_records.append(int(record_name))

        fhr_padded = np.pad(fhr, (max_signal_length - len(fhr), 0), 'constant')  # padding at the beginning
        uc_padded = np.pad(uc, (max_signal_length - len(uc), 0), 'constant')

        # Stack and append to list
        record_tensor = np.stack([fhr_padded, uc_padded], axis=0)
        tensor_data.append(record_tensor)

    # Convert the list of numpy arrays to a single numpy array
    tensor_data = np.array(tensor_data)
    # Convert to PyTorch tensor
    tensor = torch.tensor(tensor_data, dtype=torch.float32)

    return tensor, invalid_records


def filter_noisy_tensor(pytorch_tensor_processed, labels, invalid_records_names):
    """
    filters the noisy fhr,uc by locating interpolation hallucinations (fhr values below 50 or above 210 after interpolation)

    Returns: returns denoised data, and a equal in length corresponding labels tensor

    """

    # turn invalid record names into indices (CTU data), transform 0 no np.nan to isolate interpolation hallucinations.
    invalid_records_indices = [index - 1001 if index < 2000 else index - 494 - 1001 for index in
                               invalid_records_names]  # CTU record names into serial index
    pytorch_tensor_processed[pytorch_tensor_processed == 0] = np.nan
    pytorch_tensor_processed_valid_uc = np.delete(pytorch_tensor_processed, invalid_records_indices, axis=0)
    labels_uc = labels[~labels["sample"].isin(invalid_records_names)]

    # Remove outliers - fhr interpolated values below 50 or above 210
    mask_below_50 = (pytorch_tensor_processed_valid_uc[:, 0, :] <= 50).sum(axis=1) > 0
    mask_above_210 = (pytorch_tensor_processed_valid_uc[:, 0, :] >= 210).sum(axis=1) > 0
    combined_mask = mask_below_50 | mask_above_210
    labels_no_extreme_interpolated = labels_uc[~combined_mask.numpy()]
    pytorch_tensor_no_extreme_interpolated = pytorch_tensor_processed_valid_uc[~combined_mask]

    # Reindex labels by the new data shape
    labels_no_extreme_interpolated = labels_no_extreme_interpolated.copy()  # Prevent pandas warning about slicing
    labels_no_extreme_interpolated.loc[:, "old_index"] = labels_no_extreme_interpolated.index
    labels_no_extreme_interpolated_reindex = labels_no_extreme_interpolated.reset_index(drop=True)

    return pytorch_tensor_no_extreme_interpolated, labels_no_extreme_interpolated_reindex


def nanstd(o, dim, keepdim=False):
    """
    calculate std ingoring Nans

      Args:
          o: ovserved time series
          dim: dimension to calculate std
          keepdim: unsqueeze to keep dimension

    Return:
          std of the time series

    """

    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim
        )
    )

    if keepdim:
        result = result.unsqueeze(dim)

    return result


def normalize_ignor_nans(tensor, dim=1, keepdim=False):
    """
    Normalize the input tensor along the specified dimension while ignoring NaNs.

    Args:
        tensor (torch.Tensor): The input tensor to be normalized.
        dim (int): The dimension along which to perform normalization.
        keepdim (bool): Whether to keep the dimensions of the input tensor.

    Returns:
        torch.Tensor: The normalized tensor.

    """

    tensor_reshaped = tensor.view(-1, 21620)

    mean = torch.nanmean(tensor_reshaped, dim=1, keepdim=True)
    std = nanstd(tensor_reshaped, dim=1, keepdim=True)

    t_normalized = (tensor_reshaped - mean) / std
    print(t_normalized.shape)
    t_normalized = t_normalized.view(230, 2, 21620)

    return t_normalized


labels_path = "../labels.csv"
input_data_dir = "../data/ctu_chb_data"

all_records = read_records(input_data_dir)
labels = pd.read_csv(labels_path)
t, _ = create_pytorch_tensor(all_records)
t_proc, invalid_records = create_pytorch_tensor(all_records, process=True)
t_filter, l_filter = filter_noisy_tensor(t_proc, labels, invalid_records)
t_normal = normalize_ignor_nans(t_filter)

torch.save(t_filter, "../data/processed/ctg_tensor.pt")
torch.save(t_filter, "../data/processed/ctg_tensor_normalized.pt")
l_filter.to_csv("../data/processed/labels_filtered.csv")