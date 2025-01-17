import torch
import os
import pickle
import numpy as np


def convert_to_pickle(file_path, is_train=True):
    dataset = torch.load(file_path)
    dict = dataset.sim_data

    pos_data = dict['xpos_particles']
    vel_data = dict['xvel_particles']
    xfrc_ext_data = dict['xfrc_external']
    particle_mass_data = dict['particle_masses']

    if is_train:
        os.makedirs('train', exist_ok=True)
        os.makedirs('val', exist_ok=True)
        all_vels = []
        all_accs = []
    else:
        os.makedirs('test', exist_ok=True)
    
    rollouts_num, frames_num, particles_num, dim_num = pos_data.shape
    
    for i in range(rollouts_num):
        rollout_pos = pos_data[i].numpy()
        rollout_mass = particle_mass_data[i].numpy()

        rollout_mass[rollout_mass == 0] = 3
        rollout_mass[rollout_mass == 0.1] = 5
        
        # length_3 = len(rollout_mass[rollout_mass == 3])
        # length_5 = len(rollout_mass[rollout_mass == 5])
        # print(f"length_3: {length_3}, length_5: {length_5}")
        
        # convert it into int type
        rollout_mass = np.array(rollout_mass, dtype=int)
        
        rollout_mass = rollout_mass.reshape(-1)

        dict = {'position': rollout_pos, 'particle_type': rollout_mass}
        if is_train:
            if i < 100:
                pickle.dump(dict, open(f'train/{i}.pkl', 'wb'))
            else:
                pickle.dump(dict, open(f'valid/{i-100}.pkl', 'wb'))
        else:
            pickle.dump(dict, open(f'test/{i}.pkl', 'wb'))
        
        if is_train:
            # compute the mean and std of the velocity and acceleration
            # for the training data
            vel = rollout_pos[1:] - rollout_pos[:-1]
            acc = vel[1:] - vel[:-1]
            
            all_vels.append(vel)
            all_accs.append(acc)
    
    if is_train:
        all_vels = np.concatenate(all_vels, axis=0)  
        all_accs = np.concatenate(all_accs, axis=0) 
        
        all_vels_flat = all_vels.reshape(-1, 3)
        all_accs_flat = all_accs.reshape(-1, 3)
        
        vel_mean = np.mean(all_vels_flat, axis=0)
        vel_std = np.std(all_vels_flat, axis=0)
        acc_mean = np.mean(all_accs_flat, axis=0)
        acc_std = np.std(all_accs_flat, axis=0)
        
        print(f"vel_mean: {vel_mean}, vel_std: {vel_std}")
        print(f"acc_mean: {acc_mean}, acc_std: {acc_std}")


# generate train and val data
convert_to_pickle('dataset_train_full.pt', is_train=True)
# generate test data
convert_to_pickle('dataset_test.pt', is_train=False)


