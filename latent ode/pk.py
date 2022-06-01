import os
import numpy as np
import torch
import pandas as pd
import lib.utils as utils
from data_parse import read_in_data, normalize_doses, create_data_XY, random_timestep_sampling

class PK_Dynamics(object):

    timesteps = 337
    training_dimensions = 2
    doses_to_train = []
    
    n_training_samples = 100
     
    training_file = 'intensiveQD.csv'
    
    def __init__(self, root, doses, generate=False, device = torch.device("cuda:0")):
        self.root = root
        self.doses_to_train = doses
        self.n_training_samples *= len(doses)
# =============================================================================
#         if generate:
#             self._generate_dataset(doses)
# =============================================================================
    
# =============================================================================
#         if not self._check_exists():
#             raise RuntimeError('QD intensive dataset not found.')
# =============================================================================
        
        data_file = os.path.join(self.data_folder, self.training_file)
        QDcombos, combo_to_key = read_in_data(data_file)
        #QD combos only 1 combo 6
        combo_to_train_data = QDcombos[combo_to_key[1]]
        
        #combo_to_train_data = random_timestep_sampling(combo_to_train_data, doses)
        combo_to_train_data = normalize_doses(combo_to_train_data, doses)
        dataX, _ = create_data_XY(combo_to_train_data, self.doses_to_train, 3, 3)
        #print(dataX.shape)
        self.data = torch.Tensor(dataX).to(device)
        #_, self.data_min, self.data_max = utils.normalize_data(self.data)
        self.device = device
        
    
    def visualize(self, traj, plot_name = 'traj', dirname='pk_imgs', video_name = None):
        r"""Generates images of the trajectory and stores them as <dirname>/traj<index>-<t>.jpg"""
    
        T, D = traj.size()
    
        traj = traj.cpu() * self.data_max.cpu() +  self.data_min.cpu()
    
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to visualize the dataset.') from e
    
        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise Exception('PIL is required to visualize the dataset.') from e
    
        def save_image(data, filename):
            im = Image.fromarray(data)
            im.save(filename)
    
        os.makedirs(dirname, exist_ok=True)
    
        env = suite.load('pk', 'stand')
        physics = env.physics
    
        for t in range(T):
            with physics.reset_context():
                physics.data.qpos[:] = traj[t, :D // 2]
                physics.data.qvel[:] = traj[t, D // 2:]
            save_image(
                physics.render(height=480, width=640, camera_id=0),
                os.path.join(dirname, plot_name + '-{:03d}.jpg'.format(t))
            )
    
# =============================================================================
#     def _generate_dataset(self, doses):
#         if self._check_exists():
#             return
#         os.makedirs(self.data_folder, exist_ok=True)
#         print('Generating dataset...')
#         train_data = self._generate_random_trajectories(self.n_training_samples, doses)
#         torch.save(train_data, os.path.join(self.data_folder, self.training_file))
#     
#     def _generate_random_trajectories(self, n_samples, doses):
#         try:
#             from dm_control import suite  # noqa: F401
#         except ImportError as e:
#             raise Exception('Deepmind Control Suite is required to generate the dataset.') from e
#     
#         env = suite.load('pk', 'stand')
#     
#         # Store the state of the RNG to restore later.
#         st0 = np.random.get_state()
#         np.random.seed(123)
#     
#         data = np.zeros((n_samples, self.timesteps, self.training_dimensions))
#         
#         QDcombos, combo_to_key = read_in_data("intensiveQD.csv")
#         combo_to_train_data = QDcombos[combo_to_key[6]]
#         combo_to_train_data = normalize_doses(combo_to_train_data, self.doses_to_train)
#         combo_to_train_data = random_timestep_sampling(combo_to_train_data, self.doses_to_train)
#         data, _ = create_data_XY(combo_to_train_data, self.doses_to_train, 3, 3)
#     
#         # Restore RNG.
#         np.random.set_state(st0) 
#         return data
# =============================================================================
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.data_folder, self.training_file))
    
    @property
    def data_folder(self):
        return os.path.join(self.root, self.__class__.__name__)
    
    # def __getitem__(self, index):
    #     return self.data[index]
    
    def get_dataset(self):
        #print(self.data.shape)
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def size(self, ind = None):
        if ind is not None:
            return self.data.shape[ind]
        return self.data.shape
            
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

