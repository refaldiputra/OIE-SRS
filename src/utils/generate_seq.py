# A code to generate sequence ordered data set in .txt file
from typing import List
import numpy as np
import random


def create_data(max_, max_per_user):
    reg_num = max_ // max_per_user # create the base user
    data = []
    user = []
    data_irr = []
    # this loop will get up to the base user where they have the same number of items
    for i in range(reg_num):
        data.append(list(range(i*max_per_user+1, (i+1)*max_per_user+1)))
        user.append([i+1])
    # this is for the rest of the items that are not divisible by max_per_user
    if max_%max_per_user != 0: #previously there was a bug
        data_irr.append(list(range(reg_num*max_per_user+1, max_+1)))
    # concatenate the irregular data
    data = data + data_irr
    return data #data_seq

def create_file(data_seq, path):
    with open(path, 'w') as f:
        for data in data_seq:
            for d in data:
                f.write(str(d)+' ')
            f.write('\n')
            
def pick_rand(b, a=1):
    points_1 = random.sample(list(np.arange(a,b)), 2)[0]
    points_2 = random.sample(list(np.arange(a,b)), 2)[1]
    while points_1 >= points_2 or ((points_2-points_1)<3): # making sure there are at least 3 points
        points_1 = random.sample(list(np.arange(a,b)), 2)[0]
        points_2 = random.sample(list(np.arange(a,b)), 2)[1]
    return [points_1,points_2]

def generate_data(num_sample,data_seq:List):
    new_data = []
    data_seq_init = data_seq[:] # copy the value by slicing
    # this loop is for the ordered sequences
    for m in range(len(data_seq_init)): 
        data_to_insert = list(data_seq_init[m])
        data_to_insert.insert(0,m+1)
        new_data.append(data_to_insert)
    # this loop is for random sequences
    for n in range(num_sample-len(data_seq_init)): 
        # get a list in the sequence
        x_rand = random.randint(0,len(data_seq)-1)
        seq = data_seq[x_rand] # choosing a random sequence
        # generate random two points as the start and end of the sequence
        if len(seq)>2:
            points = pick_rand(len(seq))
        else: # this is the case when the sequence is less than 3.
            points = list(range(len(seq)+1))
        # append data and index
        seq_filtered = list(seq[points[0]:points[1]])
        seq_filtered.insert(0,n+1+len(data_seq)) # inserting the user_id
        new_data.append(seq_filtered)
    return new_data

# get the information from the file
def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def main():
    print('generating')
    # find the maximum item_ids in the reference data set, e.g. Beauty.txt
    name = 'LastFM'
    path_reference = #path to data .txt
    _,max_ids,_ = get_user_seqs(path_reference)
    print(f"The item size of this dataset is: {max_ids}")
    # num sample
    num_sample = 25000
    # create the sequence where the maximum per user is 20
    data_seq = create_data(max_ids, 20)
    # print('data_seq:', data_seq)
    # generate the data for file
    for_file = generate_data(num_sample,data_seq)
    path_seq = #path to new data .txt

    create_file(for_file, path_seq)
    print('done, the file is saved to', path_seq)

    # checking again
    _,max_ids,_ = get_user_seqs(path_seq)
    print(f"The item size of this dataset is: {max_ids}")

if __name__ == "__main__":
    main()
