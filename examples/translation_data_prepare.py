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
from tqdm import tqdm
from collections import OrderedDict


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


if __name__ == '__main__':
    print('translation data prepare...')
