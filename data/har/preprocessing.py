import os
import numpy as np
import json
import pandas as pd

# load a single file as a numpy array
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'data/UCI HAR Dataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'data/UCI HAR Dataset/')
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	# trainy = to_categorical(trainy)
	# testy = to_categorical(testy)
	return trainX, trainy, testX, testy

trainX, trainy, testX, testy = load_dataset()
trainX_flat = trainX.reshape((7352,1152))
testX_flat = testX.reshape((2947,1152))

train_subject_f = open('data/UCI HAR Dataset/train/subject_train.txt')
train_subject_f_lines = train_subject_f.readlines()

test_subject_f = open('data/UCI HAR Dataset/test/subject_test.txt')
test_subject_f_lines = test_subject_f.readlines()

train_subject_f_tmp = []
for line in train_subject_f_lines:
    train_subject_f_tmp.append(int(line.strip()))

train_users_list = []
for subject in train_subject_f_tmp:
    if str(subject) not in train_users_list:
        train_users_list.append(str(subject))

train_output = {}
test_output = {}

train_output_num_samples = []
test_output_num_samples = []

train_output_user_data = {}
test_output_user_data = {}

for client in train_users_list:
    train_output_user_data[client] = {}
    train_output_user_data[client]['x'] = []
    train_output_user_data[client]['y'] = []
    client_data_count = 0
    for i in range(len(train_subject_f_tmp)):
        if train_subject_f_tmp[i] == int(client):
            client_data_count += 1
            #train_output_user_data[client]['x'].append(train_X_tmp[i]+[0.0]*15)
            train_output_user_data[client]['x'].append(list(trainX_flat[i]))
            train_output_user_data[client]['y'].append(int(trainy[i][0]))
    train_output_num_samples.append(client_data_count)

test_users_list = ['testuser_1']
test_output_user_data['testuser_1'] = {}
test_output_user_data['testuser_1']['x'] = []
test_output_user_data['testuser_1']['y'] = []

test_client_data_count = 0
for i in range(len(testX)):
    test_client_data_count += 1
    #test_output_user_data['testuser_1']['x'].append(test_X_tmp[i]+[0.0]*15)
    test_output_user_data['testuser_1']['x'].append(list(testX_flat[i]))
    test_output_user_data['testuser_1']['y'].append(int(testy[i][0]))
test_output_num_samples.append(test_client_data_count)

train_output['users'] = train_users_list
train_output['num_samples'] = train_output_num_samples
train_output['user_data'] = train_output_user_data

test_output['users'] = test_users_list
test_output['num_samples'] = test_output_num_samples
test_output['user_data'] = test_output_user_data

with open('data/train/train_har.json', 'w') as outfile:
    json.dump(train_output, outfile)

with open('data/test/test_har.json', 'w') as outfile:
    json.dump(test_output, outfile)
