"""
This code is used convert single soccer log data to a numpy array
The first dimension of tensor is trajetary
the second dimension is frame
the frame follows a structure like this [status, action, reward]
for reward: goal is 1, CAPTURED_BY_DEFENSE is -1, nothing is 0
action is 5 dimension to represent Hierarchical info: the first dimension is turn, the 2, 3rd is dash, the 4th, 5th is kich
"""


# import h5py
import numpy as np
import re, os
# from policyopt import Trajectory, TrajBatch
#fileName = "soccer_traj.hdf5"
#hdf5_file = h5py.File(fileName, 'w')
def convert_log2_tensor():
	path_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'log'))#, '..','log'
	print(path_name)
	incomplete_file = open(path_name+"/incomplete.hfo")
	log_file = open(path_name+"/base_left-11.log")

	trajectory_frame_list = []
	for line in incomplete_file:

		tokens = line.split()

		if len(tokens) == 0:
			continue

		if '#' in tokens[0]:
			continue

		start = tokens[1]
		end = tokens[2]
		length = tokens[3]
		status = tokens[4]
		if 'OUT_OF_TIME' in status or 'OUT_OF_BOUNDS' in status:#skip the out of time and out of bound traj
			continue
		trajectory_frame_list.append([start, end, length, status])

	for i, x in enumerate(trajectory_frame_list):
		print(x)

	print ('the number of trajectory is:', len(trajectory_frame_list))
	# action statues, reward,

	# while trajectory_number < len(trajectory_frame_list):
	trajectory_list = []
	trajectory_number = 0
	frame_list = []
	frame = []
	num = 0
	hasReward = False
	lines = log_file.readlines()
	for i in range(0, len(lines)):
		if trajectory_number >= len(trajectory_frame_list):
			break

		start_frame = trajectory_frame_list[trajectory_number][0]
		end_frame = trajectory_frame_list[trajectory_number][1]

		tokens = lines[i].split()

		if int(tokens[0]) < int(start_frame):#skip the out of time and out of bound traj
			continue

		if len(tokens) < 4: #incomplete frame
			print("incomplete frame: ",tokens)
			break

		if (i + 1) < len(lines) and num == 2:#check duplcate action
			tokens_next_line = lines[i + 1].split()
			if 'Turn' in tokens_next_line[4] or 'Kick' in tokens_next_line[4] or 'Dash' in tokens_next_line[4] or 'Move' in tokens_next_line[4]:
				continue

		if num == 3:# 3 item in one frame
			frame = []
			num = 0

		if tokens[0] == '0' and num == 0:# for the initial action in the first frame
			if ('Turn' in tokens[4] or 'Kick' in tokens[4] or 'Dash' in tokens[4] or 'Move' in tokens[4]) and not hasReward:
				continue

		if num == 0 and tokens[0] != '0' and tokens[3] != 'GameStatus':#check any duplicates
			print('error in this line', tokens)
			break

		if tokens[0] == '0' and num == 0:#reward for the first traj int
			hasReward = True
			num = num + 1
			reward = []
			reward.append(0)
			frame.append(reward)

		elif tokens[3] == 'GameStatus' and num == 0:#reward int

			num = num + 1
			reward = []
			if int(tokens[4]) == 2:
				reward.append(-1)
			else:
				reward.append(int(tokens[4]))
			frame.append(reward)


		elif tokens[3] == 'StateFeatures' and num ==1:#status float

			num = num + 1
			status = []
			for i in range(4,len(tokens)):
				status.append(float(tokens[i]))
			frame.append(status)

		elif 'Turn' in tokens[4] and num == 2:# action turn dash kick float
			num = num + 1
			action = []
			ac = tokens[4].replace('(', ' ').replace(')', ' ').replace(',', ' ').split()

			action.append(float(ac[1]))
			action.append(0.0)
			action.append(0.0)
			action.append(0.0)
			action.append(0.0)

			frame.append(action)

			action_label = []
			action_label.append(0)
			frame.append(action_label)

		elif 'Dash' in tokens[4] and num == 2:# action turn dash kick float
			num = num + 1
			action = []
			ac = tokens[4].replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
			action.append(0.0)
			action.append(float(ac[1]))
			action.append(float(ac[2]))
			action.append(0.0)
			action.append(0.0)
			frame.append(action)

			action_label = []
			action_label.append(1)
			frame.append(action_label)

		elif 'Kick' in tokens[4] and num == 2:# action turn dash kick float
			num = num + 1
			action = []
			ac = tokens[4].replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
			action.append(0.0)
			action.append(0.0)
			action.append(0.0)
			action.append(float(ac[1]))
			action.append(float(ac[2]))
			frame.append(action)

			action_label = []
			action_label.append(2)
			frame.append(action_label)

		if num == 3:
			frame_list.append(frame)

		if int(tokens[0]) == int(end_frame) and num == 3:
			trajectory_list.append(frame_list)
			trajectory_number = trajectory_number + 1
			frame_list = []

	trajectory_list = np.array(trajectory_list)
	trajs = []
	for trajectory_ind, trajectory in enumerate(trajectory_list):
		# print('trajectory id is:',trajectory_ind)
		obs, obsfeat, actions, actiondists, rewards, action_labels = [], [], [], [], [], []
		traj_len = 0
		reward_all = 0
		for frame_id,frame in enumerate(trajectory):
			#print('frame number:', frame_id)
			reward, status, action, action_label = frame
			obs.append(status)
			actions.append(action)
			rewards.append(reward)
			action_labels.append(action_label)
			# print('status', status)
			# print('action', action)
			# print('reward', reward)
			reward_all += reward[0]
			traj_len += 1
		try:
			# print(traj_len)
			obs_T_Do = np.array(obs); assert obs_T_Do.shape == (len(obs), len(status))
		except:
			iii = 0
		obsfeat_T_Df =  obs_T_Do # just as dummpy now
		try:
			a_T_Da = np.array(actions); assert a_T_Da.shape == (len(obs), len(action))
			a_T_Da_Label = np.array(action_labels); assert a_T_Da_Label.shape == (len(obs), 1)
		except:
			iii = 0
		adist_T_Pa = a_T_Da
		r_T = np.asarray(rewards)
		try:
			r_T = r_T.ravel(); assert r_T.shape == (len(obs), )
		except:
			iii = 0
		# traj = Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, a_T_Da_Label, r_T)
		# trajs.append(traj)
		traj = {"ob":obs, "ac": actions, "ep_ret": reward_all, "re":rewards, "action_label" : action_labels}
		trajs.append(traj)
	return trajs

