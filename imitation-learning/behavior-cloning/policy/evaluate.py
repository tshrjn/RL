import torch
import pickle
import numpy as np
from torch.autograd import Variable

import data_util

def run_policy(args):

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    from policy.model import Net
    model = Net(env.observation_space.shape[0], env.action_space.shape[0])
    # Is GPU available?
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    if use_gpu:
        model=model.cuda()
    
    model.load_state_dict(torch.load(args.model))
    print("Using model: ", args.model)
    model.eval()

    latest_stat = pickle.load(open(data_util.get_latest('stats/*'),'rb'))

    for i in range(args.num_rollouts):
        print('iteration:', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = np.array(obs, dtype='float32')
            obs = data_util.normalize(obs, *latest_stat )
            if use_gpu:
                obs = torch.from_numpy(obs)
                action = model(Variable(obs.cuda(), volatile=True)).data.numpy()
            else:
                action = model(Variable(torch.from_numpy(obs), volatile=True)).data.numpy()
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))