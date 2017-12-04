import numpy as np
import pickle

np.random.seed(0)

obs = np.random.randint(100,size=(100,3))
# actions = np.array((np.sum(obs, axis=1),np.sum(np.power(obs,2),axis=1))).T
# actions = np.array((np.sum(obs, axis=1),np.sum(obs,axis=1)+4)).T
actions = np.array((np.sum(obs, axis=1),np.sum(obs,axis=1))).T

data={'actions':actions.astype(dtype='float32'),
    'observations':obs.astype(dtype='float32')}

pickle.dump(data, open('toy.pkl','wb'))
