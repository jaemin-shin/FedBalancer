import os
import numpy as np
import json
import pandas as pd

train_X_f = open('data/UCI HAR Dataset/train/X_train.txt')
train_X_f_lines = train_X_f.readlines()

train_Y_f = open('data/UCI HAR Dataset/train/y_train.txt')
train_Y_f_lines = train_Y_f.readlines()

train_subject_f = open('data/UCI HAR Dataset/train/subject_train.txt')
train_subject_f_lines = train_subject_f.readlines()

test_X_f = open('data/UCI HAR Dataset/test/X_test.txt')
test_X_f_lines = test_X_f.readlines()

test_Y_f = open('data/UCI HAR Dataset/test/y_test.txt')
test_Y_f_lines = test_Y_f.readlines()

train_X_tmp = []
for line in train_X_f_lines:
    tmp = line.strip().split(' ')
    new_tmp = []
    for item in tmp:
        if item == '':
            continue
        else:
            new_tmp.append(float(item))
    train_X_tmp.append(new_tmp)

test_X_tmp = []
for line in test_X_f_lines:
    tmp = line.strip().split(' ')
    new_tmp = []
    for item in tmp:
        if item == '':
            continue
        else:
            new_tmp.append(float(item))
    test_X_tmp.append(new_tmp)

train_Y_tmp = []
for line in train_Y_f_lines:
    train_Y_tmp.append(int(line.strip()) - 1)

test_Y_tmp = []
for line in test_Y_f_lines:
    test_Y_tmp.append(int(line.strip()) - 1)

train_subject_f_tmp = []
for line in train_subject_f_lines:
    train_subject_f_tmp.append(int(line.strip()))

label_list = []
for subject in train_Y_tmp:
    if str(subject) not in label_list:
        label_list.append(str(subject))

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
            train_output_user_data[client]['x'].append(train_X_tmp[i])
            train_output_user_data[client]['y'].append(train_Y_tmp[i])
    train_output_num_samples.append(client_data_count)

test_users_list = ['testuser_1']
test_output_user_data['testuser_1'] = {}
test_output_user_data['testuser_1']['x'] = []
test_output_user_data['testuser_1']['y'] = []

test_client_data_count = 0
for i in range(len(test_X_tmp)):
    test_client_data_count += 1
    #test_output_user_data['testuser_1']['x'].append(test_X_tmp[i]+[0.0]*15)
    test_output_user_data['testuser_1']['x'].append(test_X_tmp[i])
    test_output_user_data['testuser_1']['y'].append(test_Y_tmp[i])
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