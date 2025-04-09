import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = [] #originally this is empty
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type
        try:
            self.mode = args.mode
        except:
            self.mode = 'default'

        if self.data_type=='train': # it looks like this is the source of the autoregression2
            for user, seq in enumerate(user_seq): # enumerating from the user_seq
                input_ids = seq[-(self.max_len + 2):-2] #until up to the last two items
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1]) # here we got that the autoregression to train, that's why we have 1,2,3,4,5,5,6,7,8 becomes 1,2,3,4,5
                    self.user_ids.append(user)
        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)
        
        if self.mode == 'default':
            self.neg_answers = neg_sample
        elif self.mode == 'back':
            self.neg_answers = neg_sample_back
        elif self.mode == 'front':
            self.neg_answers = neg_sample_front

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index] # get the user-th sequence.
        input_ids = items[:-1] # get the input ids, except for the last item
        answer = items[-1] # this is the last item

        seq_set = set(items) # counts only the items without repetition
        neg_answer = self.neg_answers(seq_set, self.args.item_size) # going to a function(/)

        pad_len = self.max_len - len(input_ids) # padding according to the maximum of the length specified by the users
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:] # choosing the ids not the pad, should be okay.
        assert len(input_ids) == self.max_len # checking whether there is any error or not.

        if self.data_type in ['valid', 'test']: # only when the data set is not 'train'.
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing, this is the index that we give in the input.
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing, the real id is shifted +1 by the way.
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long), # something that is not in the S_u and selecting by random.
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set: # this means that choosing the item that is not in the item set.
        item = random.randint(1, item_size - 1)
    return item

def neg_sample_back(item_set, item_size):
    # this function will choose a random number that is lower than the minium item in the item_set
    item = random.randint(1, item_size - 1)
    if min(item_set) == 1:
        item = 2
    else:
        while item in item_set or item > min(item_set):
            item = random.randint(1, item_size - 1)
    return item

def neg_sample_front(item_set, item_size):
    # this function will choose a random number that is higher than the maximum item in the item_set
    item = random.randint(1, item_size - 1)
    if max(item_set) == item_size - 1:
        item = item_size - 2 # just before the last item, justification is to minimal the damage of disorderness
    else:
        while item in item_set or item < max(item_set):
            item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    # this is used for evaluation
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file): # this is how the data set, especially the training data set is built.
    lines = open(data_file).readlines() # it opens the file and then read it lines by lines.
    user_seq = [] # empty list
    item_set = set() # empty set
    for line in lines: # iteration
        user, items = line.strip().split(' ', 1) # this will split into two, where the first is the user then the rest is the item sequence.
        items = items.split(' ') # further splitting.
        items = [int(item) for item in items] # turns it into a list and int data type
        user_seq.append(items) # appending for the whole data set(?)
        item_set = item_set | set(items) # unionize the set (appending but no repetition)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }

    return seq_dic, max_item, num_users

def get_dataloder(args,seq_dic):

    train_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader

def random_table(max_item):
    # given the max_item we want to create a lookup table where the map will be random indices.
    # we can use numpy random, https://stackoverflow.com/questions/8505651/non-repetitive-random-number-in-numpy
    from numpy.random import default_rng
    rng = default_rng(2025) # generator class, seed for consistency
    rand_table_np = rng.choice(max_item,max_item, replace=False)
    rand_table_np +=1 #since the index starts from 1
    return rand_table_np

from dataclasses import dataclass
@dataclass
class Args:
    data_dir: str
    data_name: str

@dataclass
class Args_hparams:
    max_seq_length: int
    model_type: str

def main():
    data_dir = '/Users/refaldi/Documents/Work/llm_id_recsys/data/'
    data_name = 'LastFM'
    args = Args(data_dir,data_name)
    hparams = Args_hparams(20, 'SASRec')
    seq_dict, max_item, num_users = get_seq_dic(args)
    print(seq_dict['user_seq'][1])
    # updating the hparams
    hparams.item_size = max_item+1 # shifting the index
    hparams.num_users = num_users+1 # too
    hparams.same_target_path = os.path.join(data_dir, data_name+'_same_target.npy') # not sure what is this.
    hparams.valid_rating_matrix, hparams.test_rating_matrix = get_rating_matrix(data_name, seq_dict, max_item)

    data_train = RecDataset(hparams, seq_dict['user_seq'], data_type='train')
    data_val = RecDataset(hparams, seq_dict['user_seq'], data_type='valid')
    data_test = RecDataset(hparams, seq_dict['user_seq'], data_type='test')

    # writing them down to the .txt file.
    file_train = './train.txt'
    file_valid = './valid.txt'
    file_test = './test.txt'
    for i in range(7,15):
        print(data_train[i]) # when slicing / calling with [], we use the built-in method __getitem__
    # print(data_val[0])

    # with open(file_train, 'w+') as f:
    #     f.writelines(data_train)
    # f.close()
    print('Writing is done!')

    # rand_table_np = random_table(20)

    # print(seq_dict)
    # print(rand_table_np)

if __name__ == '__main__':
    main()
