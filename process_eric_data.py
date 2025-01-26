import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import datetime


def convert_to_pickle(file_path, output_path, is_train=True):
    dataset = torch.load(file_path)
    data_dict = dataset.sim_data

    pos_data = data_dict['xpos_particles']
    particle_mass_data = data_dict['particle_masses']
    
    os.makedirs(output_path, exist_ok=True)
    
    log_file = os.path.join(output_path, "log.txt")
    with open(log_file, 'a') as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"The current time is {current_time} \n")


    if is_train:
        os.makedirs(f"{output_path}/train", exist_ok=True)
        os.makedirs(f"{output_path}/valid", exist_ok=True)
        # Variables for incremental mean/std calculation
        sum_vel = np.zeros(3, dtype=np.float64)
        sum_vel_sq = np.zeros(3, dtype=np.float64)
        sum_acc = np.zeros(3, dtype=np.float64)
        sum_acc_sq = np.zeros(3, dtype=np.float64)
        count_vel = 0
        count_acc = 0
    else:
        os.makedirs(f"{output_path}/test", exist_ok=True)

    rollouts_num, frames_num, particles_num, dim_num = pos_data.shape
    if is_train:
        print(f"For Training Data: rollouts_num: {rollouts_num}, frames_num: {frames_num}, particles_num: {particles_num}, dim_num: {dim_num}")
        with open(log_file, "a") as f:
            f.write(f"For Training Data: rollouts_num: {rollouts_num}, frames_num: {frames_num}, particles_num: {particles_num}, dim_num: {dim_num}\n")
    else:
        print(f"For Test Data: rollouts_num: {rollouts_num}, frames_num: {frames_num}, particles_num: {particles_num}, dim_num: {dim_num}")
        with open(log_file, 'a') as f:
            f.write(f"For Test Data: rollouts_num: {rollouts_num}, frames_num: {frames_num}, particles_num: {particles_num}, dim_num: {dim_num}\n")

    for i in tqdm(range(rollouts_num)):
        print(f"Processing rollout {i}")
        
        if is_train:
            # According to Eric: training data has been processed.
            rollout_pos = pos_data[i].numpy()
        else:
            # According to Eric: test data has not been processed. This is why we need subsampling
            # convert the position to numpy array and select every 10th frame
            rollout_pos = pos_data[i].numpy()[::10]
        
        rollout_mass = particle_mass_data[i].numpy()

        rollout_mass[rollout_mass == 0] = 0
        rollout_mass[rollout_mass == 0.1] = 1
        rollout_mass = np.array(rollout_mass, dtype=int).reshape(-1)

        out_dict = {'position': rollout_pos, 'particle_type': rollout_mass}

        if is_train:
            # In Eric's data, there is no validation set. So, I just use test as valid... 
            pickle.dump(out_dict, open(f'{output_path}/train/{i}.pkl', 'wb'))
        else:
            pickle.dump(out_dict, open(f'{output_path}/valid/{i}.pkl', 'wb'))
            pickle.dump(out_dict, open(f'{output_path}/test/{i}.pkl', 'wb'))

        if is_train:
            # Compute velocities and accelerations
            vel = rollout_pos[1:] - rollout_pos[:-1]  # shape: (frames_num-1, particles_num, 3)
            # acceleration should be computed using ai = pk+1 − 2pk + pk−1
            acc = rollout_pos[2:] - 2 * rollout_pos[1:-1] + rollout_pos[:-2] # shape: (frames_num-2, particles_num, 3)

            # Flatten over frames and particles
            vel_flat = vel.reshape(-1, 3)
            acc_flat = acc.reshape(-1, 3)

            # Update counts
            count_vel += vel_flat.shape[0]
            count_acc += acc_flat.shape[0]

            # Update partial sums for mean and variance
            sum_vel += np.sum(vel_flat, axis=0)
            sum_vel_sq += np.sum(vel_flat**2, axis=0)
            sum_acc += np.sum(acc_flat, axis=0)
            sum_acc_sq += np.sum(acc_flat**2, axis=0)

    if is_train:
        # Compute means
        vel_mean = sum_vel / count_vel
        acc_mean = sum_acc / count_acc

        # Compute std
        vel_var = (sum_vel_sq / count_vel) - (vel_mean**2)
        acc_var = (sum_acc_sq / count_acc) - (acc_mean**2)
        vel_std = np.sqrt(vel_var)
        acc_std = np.sqrt(acc_var)

        print(f"vel_mean: {vel_mean}, vel_std: {vel_std}")
        print(f"acc_mean: {acc_mean}, acc_std: {acc_std}")
        
        with open(log_file, 'a') as f:
            f.write(f"vel_mean: {vel_mean}, vel_std: {vel_std}\n")
            f.write(f"acc_mean: {acc_mean}, acc_std: {acc_std}\n")
            


if __name__ == "__main__":
    
    num_rollouts = "large_ke_kd"
    
    if num_rollouts == "short":
        # config_eric_2025-01-23_01+t=2025-01-23-00-51-39 has about 32 rollouts in training
        data_path = "/arc/project/st-pai-1/se3/outputs/generate_fem_cloth_data/config_eric_2025-01-23_01+t=2025-01-23-00-51-39/"
        output_path = 'data/FEM_Eric_short'
    elif num_rollouts == "long":
        # config_eric_2025-01-21_01+t=2025-01-21-23-52-42/ has about 500 rollouts in training
        data_path = "/arc/project/st-pai-1/se3/outputs/generate_fem_cloth_data/config_eric_2025-01-21_01+t=2025-01-21-23-52-42/"
        output_path = 'data/FEM_Eric'
    elif num_rollouts == "large_ke_kd":
        data_path = "/arc/project/st-pai-1/se3/outputs/generate_fem_cloth_data/config_eric_2025-01-24_01+t=2025-01-24-19-22-12/"
        output_path = 'data/FEM_large_ke_kd'
    
    train_data = os.path.join(data_path, 'dataset_train_full.pt')
    test_data = os.path.join(data_path, 'dataset_test.pt')
   
    # Generate train and valid data
    convert_to_pickle(file_path=train_data, output_path=output_path, is_train=True)
    # Generate test data
    convert_to_pickle(file_path=test_data, output_path=output_path, is_train=False)