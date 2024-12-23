## REDD: supervised learning on the energy disaggregation problem

##### Here we test some machine learning algorithm on extracting the refrigerator energy from mains consumptions. REDD contains data of 6 houses.
#### Import packages
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import math
import os
# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#### For faster implementation, we use keras API with TensorFlow backend
from keras.utils import Sequence
from keras.layers import Input, Dense, LSTM, GRU, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
import random
from sklearn.metrics.pairwise import cosine_similarity

## PARAMETERS #################

# State: channel main 1, channel main 2, channel 3, channel 4..., channel N (energy)
# Action: percent from (main 1 + main 2): channel 3, channel 4..., channel N (%)
# with N data: (CUT_TRAIN)% train set, (100-CUT_TRAIN)% valid set

HOUSE = 1
N = 5000
CUT_TRAIN = 0.9
BACK = 5

# Plot Channels
PLT_CH = "oven"
PLT_STACK_CHS = ["dishwaser", "lighting", "washer_dryer", "refrigerator", "oven", "microwave", "electric_heat"]
PLT_COLORS = ["y", "g", "r", "c", "b", "peru", "deeppink", "coral", "gold", "turquoise", "magenta", "purple", "m", "darkkhaki", "teal", "orchid", "lawngreen", "firebrick", "darkseagreen", "cadetblue", "palevioletred", "gray"]

# AI parameters
RETRAIN = 0
EPSILON = .3 # exploration vs exploitation ratio. Here 30% exploration (random selection)
EPOCHS = 15
MAX_MEMORY = 50
BATCH = 128

# Paths

dirdata = "low_freq/"
dirmodel = "model/"

ckptpath = "{}lstmV3_qlearn_house_{}.hdf5".format(dirmodel, HOUSE)

## CHECK DIR ##################

if not os.path.isdir(dirmodel):
	os.mkdir(dirmodel)
	
## INIT #######################

hdata = {}
htime = {}
hlabel = {}

X_test = []
Y_test = []

## FUNCTIONS ##################

# load house data
def readHouse():
	data = {}
	times = {}
	labels = {}
	
	# get house's files
	for pf in os.listdir("{}house_{}".format(dirdata, HOUSE)):
	
		# channel file
		if pf.startswith("ch"):
			# channel number
			ch = int(pf.rsplit("_", 1)[1].rsplit(".")[0])

			data[ch] = {}

			# read channel file
			fr = open("{}house_{}/{}".format(dirdata, HOUSE, pf))
			for l in fr:
				l = l.strip()
				if not l:
					continue
				
				sp = l.split(" ")
				
				t = int(sp[0])
				val = float(sp[1])
				
				# set value of this channel in instant "t"
				data[ch][t] = val

		# label file
		elif pf.startswith("la"):

			fr = open("{}house_{}/{}".format(dirdata, HOUSE, pf))
			for l in fr:
				l = l.strip()
				if not l:
					continue
				
				sp = l.split(" ")
				
				ch = int(sp[0])
				label = sp[1]
				
				# add ch to the set of this label
				labels[label] = labels.get(label, [])
				labels[label].append(ch-1)

	# take data only if their time instant exists in all channels 1, 2, 3
	for t in set(data[1].keys()):
		for i in range(2, len(data)+1):
			if t not in data[i]:
				data[1].pop(t)
				break

	# get sorted times
	times[1] = list(sorted(data[1].keys()))

	return data, times, labels

# get "x & y" data of a time index "i"
def getIndDataV2(i):
	# t = time of sample number i in channel 1
	t = htime[1][i]

	# y = val of channels 3+
	y = []
	for j in range(3, len(hdata)+1):
		y.append(hdata[j][t])
	
	# x = values of channel 1 and channel 2 in time t
	x = [hdata[1][t], hdata[2][t]]

	return x, y

# fill valid set data (X & Y)
def getValidXY():
	X = []
	Y = []

	n_train = int(N * CUT_TRAIN)

	for i in range(n_train, N):
		# t = time of sample number i in channel 1
		x, y = getIndDataV2(i)

		# add to valid set
		X.append(x)
		Y.append(y)

	return X, Y

# make input sequence
def MakeInSequence(cur_state, ts, Preds):
	seq = []
	# - take mains from current state
	curmains = cur_state[:2]
	# - make sequence of [main1, main2, chan3, chan4, ..., chan20] (size 20), only main1 & main2 changes according to timestep
	for t in range(max(0, ts-BACK+1), ts+1):
		st = []
		st.extend(curmains)
		st.extend(Preds[t])
		seq.append(st)
	# - padding by last element of the sequence
	while len(seq)<BACK:
		seq.append(seq[-1])

	return seq

## LOAD DATA ##################

hdata, htime, hlabel = readHouse()
X_test, Y_test = getValidXY()

n_actions = len(hdata)-2

print("Whole Data Size: {}".format(len(hdata[1])))
print("Train Data Size: {}".format(N-len(X_test)))
print("Valid Data Size: {}".format(len(X_test)))
print("State Size: {}".format(len(hdata)))
print("Actions Size: {}".format(n_actions))

## Q LEARN ####################

X_train = []
Y_train = []
n_train = int(N * CUT_TRAIN)
n_test = N - n_train

for i in range(0, n_train):
	x, y = getIndDataV2(i)

	X_train.append(x)
	Y_train.append(y)

class Environment(object):
	# introduce and initialize all parameters and variables of the environment
	def __init__(self):
		self.total_reward_ai = 0.0
		self.total_sim_ai = 0.0
		self.total_acc_ai = 0.0
		self.cnt_energy_ai = 0

		# train or inference mode
		self.train = 1

		self.Preds = {}

	# method to update environment after AI plays an action
	def update_env(self, epercents):
		""" parameters:
		 - epercents : change energy by AI according to percentages from current (Main 1 + Main 2)
		"""
		
		if self.train == 1:
			X = X_train
			Y = Y_train
		else:
			X = X_test
			Y = Y_test

		# GETTING THE REWARD & NEW STATE

		# init: reward
		reward = 0

		# init: max real chan & Max real chan value
		maxpred = 0
		vmaxpred = 0

		# init: max real chan & Max real chan value
		maxreal = 0
		vmaxreal = 0

		# init: real percents
		real = []

		# create vector for updated state
		mains = sum(X[timestep])
		next_state = list(X[timestep])

		for i, pr in enumerate(epercents):
			# GETTING NEXT STATE
			energy = pr * mains
			next_state.append(energy)

			# Real (%)
			prr = Y[timestep][i]/mains
			real.append(prr)

			# Update max pred & real
			if prr>vmaxpred:
				maxpred = i
				vmaxpred = pr
			if prr>vmaxreal:
				maxreal = i
				vmaxreal = prr

			# GETTING THE REWARD
			reward += - abs(Y[timestep][i] - energy) / max(Y[timestep][i], energy, 1)

		# Cosine similarity between Real and Prediction (%)
		sim = cosine_similarity([epercents], [real])[0][0]

		# Accuracy (Max pred chan energy == Max real chan energy?)
		acc = int(maxpred == maxreal)

		# Total reward & Total similarity & Total accuracy
		self.total_reward_ai += reward
		self.total_sim_ai += sim
		self.total_acc_ai += acc
		self.cnt_energy_ai += 1


		# save state chans 3+ in global list (will be used for input sequence)
		self.Preds[timestep] = next_state[2:]

		return next_state
	
	# METHOD THAT RESETS THE ENVIRONMENT
	def reset(self):
		self.total_reward_ai = 0.0
		self.total_sim_ai = 0
		self.total_acc_ai = 0
		self.cnt_energy_ai = 0

		self.train = 1

	# METHOD PROVIDING INITIAL STATE
	def observe(self):
		# calc vector of current state
		if self.train == 1:
			X = X_train
			Y = Y_train
			t = 0
		else:
			X = X_test
			Y = Y_test
			t = 0

		current_state = list(X[t])

		if self.train == 1:
			# real CH 3+ energy
			current_state.extend(Y[t])
		else:
			# random CH 3+ energy
			mains = sum(X[t])
			epercents = []
			vsum = 0
			for k in range(len(hdata)-2):
				v = np.random.randint(0, 100)
				epercents.append(v)
				vsum += v
			for k in range(len(hdata)-2):
				epercents[k] /= vsum
				current_state.append(epercents[k] * mains)

		# save state chans 3+ in global list (will be used for input sequence)
		self.Preds[t] = current_state[2:]

		return current_state

class Brain(object):
	def __init__(self, learning_rate, n_actions):

		# Input layer
		states = Input(shape = (BACK, len(hdata)))

		# LSTM layer
		x = LSTM(32)(states)

		# 2 Fully Connected (Dense) layers
		x = Dense(units = 32, activation = 'relu')(x)
		x = Dense(units = 32, activation = 'relu')(x)

		# Final Fully Connected (Dense) layer (Output)
		q_values = Dense(units = n_actions, activation = 'softmax')(x)

		# Attach layers to model & loss function & optimizer
		self.model = Model(inputs = states, outputs = q_values)
		self.model.compile(loss='mse', optimizer = Adam(lr = learning_rate))

class DQN(object):
	
	# INITIALIZE ALL THE PARAMETERS AND VARIABLES OF THE DQN
	def __init__(self):
		self.memory = list()

	# METHOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY
	def remember(self, transition):
		"""arguments:
		transition: tuple of 2 elemnts (current state, time step)
		"""

		self.memory.append(transition)

		if len(self.memory) > MAX_MEMORY:
			# delete first memory element (FIFO)
			del self.memory[0]
	
	# CONSTRUCT BATCHES OF INPUTS AND TARGETS BY EXTRACTING TRANSITIONS FROM THE MEMORY
	def get_batch(self, Preds):
		# memory size
		len_memory = len(self.memory)
		# select first elmnt of transition tuple, ie shape of state vector
		num_inputs = len(self.memory[0][0])
		
		# initialize the batches
		inputs = np.zeros((min(len_memory, BATCH), BACK, num_inputs))   # typically BATCH x BACK x num inputs
		targets = np.zeros((min(len_memory, BATCH), n_actions)) # typically BATCH x num actions
		
		# extract random transitions from memory and populate input states and outputs Q-values
		for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, BATCH))):
		
			current_state, ts = self.memory[idx]
			
			# take input sequence
			seq = MakeInSequence(current_state, ts, Preds)

			inputs[i] = np.array(seq)

			mains = sum(X_train[ts])
			for k in range(len(Y_train[ts])):
				realpr = Y_train[ts][k] / mains
				targets[i][k] = realpr

		return inputs, targets

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)

if RETRAIN or not os.path.exists(ckptpath):
	# BUILDING THE ENVIRONMENT BY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
	env = Environment()

	# BUILDING THE NEURAL NETWORK OBJECT USING BRAIN CLASS
	brain = Brain(learning_rate = 0.0001, n_actions = n_actions)

	# BUILDING THE DQN MODEL
	dqn = DQN()

	# CHOOSING THE MODE
	env.train = True

	# TRAINING THE AI
	model = brain.model

	# (+) SHOW MODEL ARCHITECTURE INFORMATIONS
	model.summary()

	# STARTING THE LOOP OVER ALL THE EPOCHS
	for epoch in range(0, EPOCHS):
		
		# INITIALIZING ALL THE VARIABLES OF BOTH THE ENVIRONMENT AND THE TRAINING LOOP
		env.reset()
		current_state = env.observe()
		
		# STARTING THE LOOP OVER ALL THE TIMESTEPS IN ONE EPOCH
		for timestep in range(1, len(Y_train)):
			
			print("Epoch {} Step {}       ".format(epoch+1, timestep), end="\r", flush=True)

			# PLAYING THE NEXT ACTION BY EXPLORATION (CHANCE: EPSILON)
			if np.random.rand() <= EPSILON:

				epercents = []
				vsum = 0
				for k in range(len(hdata)-2):
					v = np.random.randint(0, 100)
					epercents.append(v)
					vsum += v
				for k in range(len(hdata)-2):
					epercents[k] /= vsum

			# PLAYING THE NEXT ACTION BY INFERENCE
			else:
				# make input sequence
				seq = MakeInSequence(current_state, timestep-1, env.Preds)

				# make prediction to play action
				epercents = model.predict(np.array([seq]))[0]

			print("Epoch {} Step {}: update env...       ".format(epoch+1, timestep), end="\r", flush=True)

			# UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
			next_state = env.update_env(epercents)
			
			# STORING THIS NEW TRANSITION INTO THE MEMORY
			dqn.remember([current_state, timestep])
			
			print("Epoch {} Step {}: get batch...       ".format(epoch+1, timestep), end="\r", flush=True)

			# GATHERING IN TWO SEPARATE BATCHES THE INPUTS AND THE TARGETS
			inputs, targets = dqn.get_batch(env.Preds)

			print("Epoch {} Step {}: train on batch...       ".format(epoch+1, timestep), end="\r", flush=True)

			# COMPUTING THE LOSS OVER THE TWO WHOLE BATCH OF INPUTS AND TARGETS
			# training the model on the batch
			model.train_on_batch(inputs, targets)
			# update the current state
			current_state = next_state					 
			timestep += 1

		# PRINTING THE TRAINING RESULTS FOR EACH EPOCH
		print("\nEpoch: {:03d}/{:03d}".format(epoch+1, EPOCHS))
		print("Total Rewards AI: {}".format(env.total_reward_ai/env.cnt_energy_ai))
		print("Total Similarity AI: {}".format(env.total_sim_ai/env.cnt_energy_ai))
		print("Total Accuracy AI: {}".format(env.total_acc_ai/env.cnt_energy_ai))
		
		# SAVING THE MODEL
		model.save(ckptpath)
else:
	# OR JUST LOAD PRE-TRAINED MODEL
	model = load_model(ckptpath)

	# (+) SHOW MODEL ARCHITECTURE INFORMATIONS
	model.summary()

print('Evaluating...')

# BUILDING THE ENVIRONMENT BY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = Environment()

# CHOOSING THE MODE
env.train = False

# RUNNING SIMULATION INFERENCE MODE
current_state = env.observe()

''' (+) Begin [plot] '''
if PLT_CH not in hlabel:
	PLT_CH = "lighting"
chans = hlabel[PLT_CH]
lmains = []
ltemps = []
lreal = []
lpred = []
lpred_chans = {}
color_chans = {}
for i, label in enumerate(PLT_STACK_CHS):
	if label in hlabel:
		lpred_chans[label] = []
		color_chans[label] = PLT_COLORS[i]
''' (+) End [plot] '''

# STARTING THE LOOP OVER TESTING TIMESTEPS
for timestep in range(1, len(X_test)):
	print("Evaluate: {}/{}".format(timestep, len(X_test)-1), flush=True, end="\r")

	# make input sequence
	seq = MakeInSequence(current_state, timestep-1, env.Preds)

	epercents = model.predict(np.array([seq]))[0]
	
	# UPDATING ENVIRONMENT AND REACHING THE NEXT STATE
	next_state = env.update_env(epercents)
	# update the current state
	current_state = next_state
	
	''' (+) Begin [plot] '''
	# time steps
	ltemps.append(timestep)
	
	# house real & pred chans 3+
	sreal = 0
	spred = 0
	for ch in chans:
		sreal += Y_test[timestep][ch]
		spred += current_state[ch+2]
	lreal.append(sreal)
	lpred.append(spred)

	# house mains
	mains = sum(current_state[:2])
	lmains.append(mains)

	# house chosen chans
	sumchans = 0
	for label in PLT_STACK_CHS:
		if label in hlabel:
			energy = sum(map(lambda ch: current_state[ch], hlabel[label]))
			lpred_chans[label].append(energy)
			sumchans += energy
	''' (+) End [plot] '''

# (+) PRINTING THE RESULTS FOR 1 YEAR
print("Total Rewards AI: {}".format(env.total_reward_ai/env.cnt_energy_ai))
print("Total Similarity AI: {}".format(env.total_sim_ai/env.cnt_energy_ai))
print("Total Accuracy AI: {}".format(env.total_acc_ai/env.cnt_energy_ai))

''' (+) Begin [plot] '''
# Stack bar
plt.title("Energy Disaggregation Histogram", fontsize=25)
plt.xlabel("Time", fontsize=20)

labels = ["Mains"]
bottom = 0
for key in lpred_chans.keys():
	lpred_chans[key] = np.array(lpred_chans[key])
	
	label = key[0].upper() + key[1:]
	labels.append(label)

	plt.bar(ltemps, lpred_chans[key], label=label, bottom=bottom, color=color_chans[key])
	
	bottom += lpred_chans[key]

plt.plot(ltemps, lmains)

plt.legend(labels=labels)
plt.show()

# Curve
plt.figure()
plt.title("Energy Disaggregation Channel {}".format(PLT_CH), fontsize=25)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Energy", fontsize=20)
plt.plot(ltemps, lreal, "g", label="Real Energy")
plt.plot(ltemps, lpred, "b", label="Predicted Energy")
plt.legend()
plt.show()
''' (+) End [plot] '''
