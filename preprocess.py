
import pandas as pd
import time
import math
import pickle
import argparse
import os



def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    column_dict2 = {i: x for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict, column_dict2


def convert_time(df, column_name, start_time, end_time):
    start_struct_time = time.strptime(start_time+' 00:00:00', "%Y-%m-%d %H:%M:%S")
    start_unix_time = int(time.mktime(start_struct_time))
    end_struct_time = time.strptime(end_time+' 23:59:59', "%Y-%m-%d %H:%M:%S")
    end_unix_time = int(time.mktime(end_struct_time))

    df[column_name] = df[column_name].apply(lambda x:(x-start_unix_time)/100)

    last_user = ''
    for index, row in df.iterrows():
        if index == 0:
            continue
        if df.loc[index, 'user'] != last_user:
            last_user = df.loc[index, 'user']
            continue
        while df.loc[index, column_name] <= df.loc[index-1, column_name]:
            df.loc[index, column_name]+=1


    return df


def create_user_list(df, user_size=None, item_size=None):
    if user_size == None:
        user_size = len(df['user'].unique())
    if item_size == None:
        item_size = len(df['item'].unique())

    user_list = [list() for u in range(user_size)]
    for row in df.itertuples():
        if row.rating > 3.0:
            user_list[row.user].append((row.item, row.timestamp))
        else:
            user_list[row.user].append((row.item + item_size, row.timestamp))
    return user_list


def split_train_test(df, user_size=None, test_size=0.2):
    if user_size == None:
        user_size = len(df['user'].unique())

    total_user_list = create_user_list(df, user_size)
    train_user_list = [None] * user_size
    test_user_list = [None] * user_size
    for user, item_list in enumerate(total_user_list):
        item_list = sorted(item_list, key=lambda x: x[1])
        test_item = item_list[math.floor(len(item_list)*(1-test_size)):]
        train_item = item_list[:math.floor(len(item_list)*(1-test_size))]
        test_user_list[user] = test_item
        train_user_list[user] = train_item

    return train_user_list, test_user_list


def main(args):
    data_path = '../../Data/'

    data = pd.read_csv(data_path + args.dataset + '.csv')
    data, user_mapping, user_mapping2 = convert_unique_idx(data, 'user')
    data, item_mapping, item_mapping2 = convert_unique_idx(data, 'item')
    print('Complete assigning unique index to user and item')

    data = convert_time(data, 'timestamp', args.start_time, args.end_time)
    print('Complete transformation for timestamp')
    # print(data.head(30))

    user_size = len(data['user'].unique())
    item_size = len(data['item'].unique())

    train_user_list, test_user_list = split_train_test(data,
                                                       user_size,
                                                       test_size=args.test_size)
    print('Complete spliting items for training and testing')

    dataset = {'user_size': user_size, 'item_size': item_size,
               'user_mapping': user_mapping, 'item_mapping': item_mapping,
               'train_user_list': train_user_list, 'test_user_list': test_user_list}

    with open(data_path + args.dataset + '.pickle', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Complete preprocessing')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        help="Name of dataset to be preprocessed")
    parser.add_argument('--start_time',
                        type=str,
                        default='2014-01-01')
    parser.add_argument('--end_time',
                        type=str,
                        default='2014-06-30')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help="Proportion for training and testing split")
    args = parser.parse_args()
    main(args)


