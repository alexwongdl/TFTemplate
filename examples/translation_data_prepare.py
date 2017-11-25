# coding=utf-8
"""
Created by Alex Wang
On 2017-11-19

translation data prepare:
1.nltk tokenize
2.build vocabulary with most frequency words, like:
    _PAD
    _GO
    _EOS
    _UNK
    the
    .
    ,
    of
    and
    to
    in
    a
    )
    (
    0
    for
    00
    that
    is
    on
3. transfer sententces into number representation
"""
import os
import json
import nltk
from myutil import pathutil


# nltk.download('punkt') # in case exception 'Failed loading english.pickle with nltk.data.load' occurs

def prepare_data(corpora_one, corpora_two, corpora_combine, dic_one_path, dic_two_path, corpora_combine_ID):
    """
    :param corpora_one:
    :param corpora_two:
    :param corpora_combine: json representation {'source':'Il a transform¨¦ notre vie','target':'Changing Lives'}
    :param dic_one_path: path to save dictionary for corpora_one
    :param dic_two_path: path to save dictionary for corpora_two
    :param corpora_combine_ID: ID representation of corpora_combine
    :return:
    """
    if os.path.isfile(dic_one_path):
        print('{} already exits.'.format(dic_one_path))
        return

    word_count_one = dict()
    word_count_two = dict()

    with open(corpora_one, 'r') as reader_one, open(corpora_two, 'r') as reader_two, \
            open(corpora_combine, 'w') as combine_writer:

        line_one = reader_one.readline().lower()
        line_two = reader_two.readline().lower()

        # test
        # tokens_one = nltk.word_tokenize(line_one)
        # tokens_two = nltk.word_tokenize(line_two)
        # for token in tokens_one:
        #     count = word_count_one[token] + 1 if token in word_count_one.keys() else 1
        #     word_count_one[token] = count
        # for token in tokens_two:
        #     count = word_count_two[token] + 1 if token in word_count_two.keys() else 1
        #     word_count_two[token] = count
        # combine_writer.write(json.dumps({'source': tokens_one, 'target': tokens_two}) + '\n')

        # run all
        while line_one:
            tokens_one = nltk.word_tokenize(line_one)
            tokens_two = nltk.word_tokenize(line_two)

            for token in tokens_one:
                count = word_count_one[token] + 1 if token in word_count_one.keys() else 1
                word_count_one[token] = count

            for token in tokens_two:
                count = word_count_two[token] + 1 if token in word_count_two.keys() else 1
                word_count_two[token] = count

            combine_writer.write(json.dumps({'source': tokens_one, 'target': tokens_two}) + '\n')
            line_one = reader_one.readline().lower()
            line_two = reader_two.readline().lower()

    sorted_one = sorted(word_count_one.items(), key=lambda x: x[1], reverse=True)
    dict_one = ['_PAD', '_GO', '_EOS', '_UNK']
    dict_one.extend([x[0] for x in sorted_one])
    dict_one = dict_one[0:40000]

    sorted_two = sorted(word_count_two.items(), key=lambda x: x[1], reverse=True)
    dict_two = ['_PAD', '_GO', '_EOS', '_UNK']
    dict_two.extend([x[0] for x in sorted_two])
    dict_two = dict_two[0:40000]

    with open(dic_one_path, 'w') as writer:
        [writer.write(word + "\n") for word in dict_one]
    with open(dic_two_path, 'w') as writer:
        [writer.write(word + "\n") for word in dict_two]


def corpora_to_id(corpora_one, corpora_two, corpora_combine, dic_one_path, dic_two_path, corpora_combine_ID):
    """
    语料转化成id表示
    :param corpora_one:
    :param corpora_two:
    :param corpora_combine:
    :param dic_one_path:
    :param dic_two_path:
    :param corpora_combine_ID:
    :return:
    """
    fr_word_to_id, fr_id_to_word = load_dict(dic_one_path)
    en_word_to_id, en_id_to_word = load_dict(dic_two_path)
    count = 0
    with open(corpora_combine, 'r') as reader, open(corpora_combine_ID, 'w') as writer:
        line = reader.readline()
        while line:
            count += 1
            strings_dict = json.loads(line)
            tokens_fr = strings_dict['source']
            tokens_en = strings_dict['target']
            tokens_fr_id = [fr_word_to_id[word.strip()] if word.strip() in fr_word_to_id.keys() else 3 for word in tokens_fr]
            tokens_en_id = [en_word_to_id[word.strip()] if word.strip() in en_word_to_id.keys() else 3 for word in tokens_en]
            writer.write(json.dumps({'source':tokens_fr_id, 'target':tokens_en_id}) + '\n')
            if count < 10:
                words_fr = [fr_id_to_word[id] for id in tokens_fr_id]
                words_en = [en_id_to_word[id] for id in tokens_en_id]
                print('example:' + str(count))
                print('corpora SOURCE:{}, TARGET:{}'.format(' '.join(tokens_fr), ' '.join(tokens_en)))
                print('corpora id SOURCE:{}, TARGET:{}'.format(' '.join('{}'.format(id) for id in tokens_fr_id), ' '.join('{}'.format(id) for id in tokens_en_id)))
                print('id_to_word SOURCE:{}, TARGET:{}'.format(' '.join(words_fr), ' '.join(words_en)))
                print('')

            line = reader.readline()


def load_dict(dict_path):
    """
    加载字典
    :param dict_path:
    :return:
    """
    word_to_id = {}
    id_to_word = {}
    index = 0
    print('start load dict from {}'.format(dict_path))
    with open(dict_path, 'r') as reader:
        word = reader.readline()
        while word:
            word = word.strip()
            word_to_id[word] = index
            id_to_word[index] = word
            index += 1
            word = reader.readline()

    print('size of word_to_id:{}'.format(len(word_to_id)))
    print('size of id_to_word:{}'.format(len(id_to_word)))
    for i in range(10):
        print('{}:{}'.format(i, id_to_word[i]))
    return word_to_id, id_to_word

def load_source_target_id_corpus(file_path):
    """
    加载id表示的语料，每一行是{'source':xxx, 'target':ooo}
    :param file_path:
    :return:
    """
    string_pair_list = []
    with open(file_path, 'r') as reader:
        line = reader.readline()
        while line:
            string_pair = json.loads(line)
            string_pair_list.append(string_pair)
            line = reader.readline()
    return string_pair_list

def load_sub_files(file_dir):
    """
    获取训练文件子文件
    :param file_dir:
    :return:
    """
    file_obs_list, file_list = pathutil.list_files(file_dir)
    sub_files_obs = []
    sub_files_rel = []
    for file_obs, file_rel in zip(file_obs_list, file_list):
        if 'id' in file_rel:
            sub_files_obs.append(file_obs)
            sub_files_rel.append(file_rel)
    return sub_files_obs, sub_files_rel

def load_train_data(file_path):
    """
    加载训练数据
    :param file_path:
    :return:
    """
    print("load " + file_path)
    data_list = []
    with open(file_path, 'r') as reader:
        line = reader.readline()
        while line:
            ids_dict = json.loads(line)
            data_list.append(ids_dict)
            line = reader.readline()
    print('load {} records .'.format(len(data_list)))
    for i in range(3):
        print(json.dumps(data_list[i]))
    return data_list

def format_data(file_path, batch_size):
    """
    加载训练数据，整理成batch_size的形式
    :param file_path:
    :param batch_size:
    :return: x_batch_list, y_batch_list, target_list,  x_length_list, y_length_list, target_weight_list
    """
    print('format file: ' + file_path)
    _PAD =  0
    _GO = 1
    _EOS = 2

    data_list = load_train_data(file_path)
    data_batch = []

    batch_num = len(data_list) // batch_size
    for i in range(batch_num):
        temp_data_list = data_list[i*batch_size : (i+1)*batch_size]
        temp_x_list, temp_y_list, temp_target_list, temp_x_len_list, temp_y_len_list, temp_target_weight_list = [],[],[],[],[],[]
        for data in temp_data_list:
            temp_x_len_list.append(len(data['source']))
            temp_y_len_list.append(len(data['target']) + 1) # first item is <_GO>
        max_x_len = max(temp_x_len_list)
        max_y_len = max(temp_y_len_list)
        for data in temp_data_list:
            len_source =  len(data['source'])
            len_target = len(data['target'])

            temp_x = data['source']
            temp_x.extend([_PAD] * (max_x_len - len_source))

            temp_y = [_GO]
            temp_y.extend(data['target'])
            temp_y.extend([_PAD] * (max_y_len - 1 - len_target))

            temp_target = data['target']
            temp_target.extend([_EOS])
            temp_target.extend([_PAD] * (max_y_len - 1 - len_target))

            temp_target_weight = [1.0] * (len_target + 1)
            temp_target_weight.extend([0.0] * (max_y_len - len_target - 1))

            temp_x_list.append(temp_x)
            temp_y_list.append(temp_y)
            temp_target_list.append(temp_target)
            temp_target_weight_list.append(temp_target_weight)

        info = {'x_list':temp_x_list, 'y_list':temp_y_list, 'target_list':temp_target_list,
                'x_len_list':temp_x_len_list, 'y_len_list':temp_y_len_list, 'target_weight':temp_target_weight_list}

        data_batch.append(info)
    return data_batch

def format_files(input_dir, output_dir, batch_size, source_dict_path, target_dict_path ):
    """
    输入文件格式化后输出到output_dir
    :param input_dir:
    :param output_dir:
    :param batch_size:
    :return:
    """
    _, source_id_to_word = load_dict(source_dict_path)
    _, target_id_to_word = load_dict(target_dict_path)

    sub_files_obs, sub_files_rel = load_sub_files(input_dir)
    for sub_file_obs, sub_file_rel in zip(sub_files_obs, sub_files_rel):
        data_batch = format_data(sub_file_obs, batch_size)
        count = 0
        with open(os.path.join(output_dir, sub_file_rel) , 'w') as writer:
            for data in data_batch:
                writer.write(json.dumps(data) + '\n')
                count += 1
                if count < 3:
                    print('x_list:')
                    print(batch_id_to_word(data['x_list'], source_id_to_word))
                    print('x_len_list:')
                    print(data['x_len_list'])
                    print('y_list:')
                    print(batch_id_to_word(data['y_list'], target_id_to_word))
                    print('y_len_list:')
                    print(data['y_len_list'])
                    print('target_list:')
                    print(batch_id_to_word(data['target_list'], target_id_to_word))
                    print('target_weight:')
                    print(data['target_weight'])
                    print("--------------------------------------------------------")

def batch_id_to_word(id_list_list, id_to_word_dict):
    result = []
    for id_list in id_list_list:
        temp_result = [id_to_word_dict[id] for id in id_list]
        result.append(temp_result)
    return result


if __name__ == '__main__':
    print('translation data prepare...')
