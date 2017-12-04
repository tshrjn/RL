import pickle
import numpy as np
from torch.utils.data import Dataset

class ExpertEnvDataset(Dataset):
    """docstring for ExpertEnvDataset"""
    def __init__(self, data, stat=None):
        self.observations = data['observations']
        self.actions = data['actions']
        # self.returns =self.normalize( data['returns'])
        assert (len(self.observations) == len(self.actions))

        self.in_shape = data['observations'].shape[1:]
        self.out_shape = data['actions'].shape[1:]

        self.mean = np.mean(data['observations'], axis=0)
        self.std = np.std(data['observations'], axis=0)

        if stat == None:
            self.observations = normalize(self.observations, self.mean, self.std)
        else:
            self.observations = normalize(self.observations, *stat)

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        # sample_dict = {
        #         'observation': self.observations[idx].flatten(),
        #         'action': self.actions[idx].flatten(),
                # 'returns': self.returns[idx]
                # }

        sample = (self.observations[idx].flatten(), 
                self.actions[idx].flatten())

        # if self.transform:
            # sample = self.transform(sample)

        return sample

    def dimensions(self):
        '''returns (flattened in_shape, flattened out_shape)'''
        return (len(self.observations[1].flatten()), 
            len(self.actions[1].flatten()) )

    def original_dimensions(self):
        '''returns (in_shape, out_shape)'''
        return self.in_shape, self.out_shape


def normalize(data, mean, std):
    return (data - mean)/(std + 1e-6)


def process_data(args):
    data = pickle.load(open(args.out, 'rb'))
    if args.small:
        idx = np.random.randint(len(data['actions']), size=100)
        data = {k: v[idx,:] for (k,v) in data.items()}

    train_data, val_data = split(data, args.val_ratio)

    train_data = ExpertEnvDataset(train_data)
    val_data = ExpertEnvDataset(val_data, (train_data.mean, train_data.std))

    pickle.dump((train_data.mean, train_data.std), open('stats/train.pth','wb'))
    return train_data, val_data


def split(data, val_ratio):
    val_idx = int(len(data['actions'])*(1-val_ratio))

    print("Split:", val_idx, len(data['actions']))
    train_data = {key: data[key][:val_idx] for key in data.keys()}
    val_data = {key: data[key][val_idx:] for key in data.keys()}

    return train_data, val_data


def get_latest(directory):
    import glob
    import os

    list_of_files = glob.glob(directory.strip(' /*') + '/*')
    latest = max(list_of_files, key=os.path.getctime)

    return latest