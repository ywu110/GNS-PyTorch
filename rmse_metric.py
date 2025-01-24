import numpy as np
from typing import Dict
import sklearn.metrics as metrics
import os
import pickle
import argparse


def mse_signal(
    y_pred: np.ndarray,
    y_gt: np.ndarray,
    rmse_or_mse: str="mse") -> Dict[str, np.ndarray]:
    r"""
    Compute MSE/RMSE of predicted vs GT signal. 1) framewise (r)mse of
    multiple motion clips; 2) per-clip (r)mse as the average of framewise
    mse across steps; 3) per-step (r)mse as the average of framewise mse
    across clips; 4) all-clips (r)mse as the average of per-clip mse.

    Parameters:
        y_pred: Predicted values
            (num_clips, clip_length, num_particles, motion_dims)
        y_gt: Ground truth values
            (num_clips, clip_length, num_particles, motion_dims)
        rmse_or_mse: "rmse" or "mse"

    Return:
        A dictionary containing:
        - framewise_err OR framewise_rmse: (num_clips, clip_length, 1)
        - per_clip_err OR per_clip_rmse: (num_clips, 1)
        - per_frame_err OR per_frame_rmse: (clip_length, 1)
        - all_clips_err OR all_clips_rmse, including 1) mean and 2) std:
            (2, )
    """
    # input checks
    assert y_pred.shape == y_gt.shape, \
        "Shape mismatch between y_pred and y_gt"
    assert rmse_or_mse in ["rmse", "mse"], \
        "Unsupported rmse_or_mse value"
    
    num_clips, clip_length, _, _ = y_pred.shape

    # compute step-wise mse
    framewise_err = np.zeros(
        (num_clips, clip_length, 1), dtype=np.float32)
    for clip_idx in range(num_clips):
        for step_idx in range(clip_length):
            rmse = metrics.root_mean_squared_error(
                y_gt[clip_idx, step_idx].flatten(),
                y_pred[clip_idx, step_idx].flatten())
            if rmse_or_mse == "rmse":
                framewise_err[clip_idx, step_idx] = rmse
            else:
                framewise_err[clip_idx, step_idx] = np.square(rmse)
    
    # compute per-clip error
    per_clip_err = np.mean(
        framewise_err, axis=1).reshape(num_clips, 1)
    
    # compute per-step error
    per_frame_err = np.mean(
        framewise_err, axis=0).reshape(clip_length, 1)
    
    # compute all-clips error, including mean and std
    all_clips_err_mean = np.mean(framewise_err)
    all_clips_err_std = np.std(framewise_err)
    all_clips_err = np.array(
        [all_clips_err_mean, all_clips_err_std]).reshape(2, )

    return {
        f"framewise_{rmse_or_mse}": framewise_err,
        f"per_clip_{rmse_or_mse}": per_clip_err,
        f"per_frame_{rmse_or_mse}": per_frame_err,
        f"all_clips_{rmse_or_mse}": all_clips_err}


def mse_param(
    param_pred: np.ndarray,
    param_gt: np.ndarray,
    rmse_or_mse: str="mse") -> Dict[str, np.ndarray]:
    r"""
    Compute (R)MSE of an N-dimensional parameter.

    Parameters:
        param_pred: Predicted values (num_params, 1)
        param_gt: Ground truth values (num_params, 1)
        rmse_or_mse: "rmse" or "mse"

    Return:
        A dictionary containing:
        - param_err OR param_rmse: (1)
    """
    rmse = metrics.root_mean_squared_error(
        param_gt.flatten(),
        param_pred.flatten())
    if rmse_or_mse == "rmse":
        err = rmse
    else:
        err = np.square(rmse)
    return {f"param_{rmse_or_mse}": err}

def form_ndarray(dir_path:str):
    # load all the pickle files under the directory
    all_data = []
    for file in os.listdir(dir_path):
        if file.endswith(".pkl"):
            with open(os.path.join(dir_path, file), 'rb') as f:
                pickle_data = pickle.load(f)
                all_data.append(pickle_data)
    
    # for each pickle file, extract the necessary data
    data_gt = []
    data_pred = []
    
    for data in all_data:
        data_gt.append(data['tgt_poss'])
        data_pred.append(data['pred_poss'])
    
    return np.array(data_gt), np.array(data_pred)

def main():
    # read from args to get the directory path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True, help="The name of directory containing the pickle files")
    args = parser.parse_args()
    
    dir_path = args.dir_path
    data_gt, data_pred = form_ndarray(dir_path)
    
    print(f"The shape of data_gt is {data_gt.shape}")
    print(f"The shape of data_pred is {data_pred.shape}")
    
    # compute the mse for the data
    metric_dict = mse_signal(data_pred, data_gt, "rmse")
    
    # save the metric_dict to a file
    with open(os.path.join(dir_path, "metric_dict.pkl"), 'wb') as f:
        pickle.dump(metric_dict, f)
    
    # print the mean and std of the all_clips_rmse
    print(f"The mean of all_clips_rmse is {metric_dict['all_clips_rmse'][0]}")
    print(f"The std of all_clips_rmse is {metric_dict['all_clips_rmse'][1]}")

if __name__ == "__main__":
    main()
