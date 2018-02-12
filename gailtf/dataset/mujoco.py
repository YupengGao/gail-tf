from gailtf.baselines import logger
import pickle as pkl
import numpy as np
from tqdm import tqdm
import ipdb

class Dset(object):
    def __init__(self, inputs, labels_low, labels_high, randomize):
        self.inputs = inputs
        self.labels_high = labels_high
        self.labels_low = labels_low
        assert len(self.inputs) == len(self.labels_low)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()
       
    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels_low = self.labels_low[idx, :]
            self.labels_high = self.labels_high[idx, :]

    def get_next_batch(self, batch_size, isHigh):
        # if batch_size is negative -> return all
        if batch_size < 0:
            if isHigh:
                return self.inputs, self.labels_high
            else:
                return self.inputs, self.labels_low
            # return self.inputs, self.labels_low, self.labels_high
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels_low = self.labels_low[self.pointer:end, :]
        labels_high = self.labels_high[self.pointer:end, :]
        self.pointer = end
        assert len(labels_low) == len(labels_high)
        if isHigh:
            return inputs, labels_high
        else:
            return inputs, labels_low
        # return inputs, labels_low, labels_high

class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, ret_threshold=None, traj_limitation=np.inf, randomize=True):
        # with open(expert_path, "rb") as f:
        #     traj_data = pkl.load(f)
        from gailtf.common import convert_log2_tensor
        traj_data = convert_log2_tensor.convert_log2_tensor()
        obs = []
        # acs = []
        rets = []
        lens = []
        actions_high = []
        actions_low = []
        for traj in tqdm(traj_data):
            if ret_threshold is not None and traj["ep_ret"] < ret_threshold:
                pass
            if len(rets) >= traj_limitation:
                break
            rets.append(traj["ep_ret"])
            lens.append(len(traj["ob"]))
            obs.append(traj["ob"])
            actions_low.append(traj["actions_low"])
            actions_high.append(traj["actions_high"])
        self.num_traj = len(rets)
        self.avg_ret = sum(rets)/len(rets)
        self.avg_len = sum(lens)/len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        self.obs = np.array([v for ob in obs for v in ob])
        self.acs_low = np.array([v for ac in actions_low for v in ac])
        self.acs_high = np.array([v for ac in actions_high for v in ac])
        if len(self.acs_low) > 2:
            self.acs_low = np.squeeze(self.acs_low)
        assert len(self.obs) == len(self.acs_low)
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs_low, self.acs_high, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction),:], 
                              self.acs_low[:int(self.num_transition*train_fraction),:],
                              self.acs_high[:int(self.num_transition*train_fraction),:],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):,:], 
                            self.acs_low[int(self.num_transition*train_fraction):,:],
                            self.acs_high[int(self.num_transition * train_fraction):, :],
                            self.randomize)
        self.log_info()
        print("finish load data!")

    def log_info(self):
        logger.log("Total trajectories: %d"%self.num_traj)
        logger.log("Total transitions: %d"%self.num_transition)
        logger.log("Average episode length: %f"%self.avg_len)
        logger.log("Average returns: %f"%self.avg_ret)

    def get_next_batch(self, batch_size, split=None, isHigh=False):
        if split is None:
            return self.dset.get_next_batch(batch_size, isHigh)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size, isHigh)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size, isHigh)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path):
    dset = Mujoco_Dset(expert_path)
    dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../baselines/ppo1/ppo.Hopper.0.00.pkl")
    args = parser.parse_args()
    test(args.expert_path)

