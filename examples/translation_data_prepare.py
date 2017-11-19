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

import json
import nltk
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
    word_count_one = dict()
    word_count_two = dict()

    with open(corpora_one, 'r') as reader_one, open(corpora_two, 'r') as reader_two, \
            open(corpora_combine,'w') as combine_writer:

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

    sorted_two = sorted(word_count_two.items(), key=lambda x: x[1], reverse=True)
    dict_two = ['_PAD', '_GO', '_EOS', '_UNK']
    dict_two.extend([x[0] for x in sorted_two])

    with open(dic_one_path, 'w') as writer:
        [writer.write(word + "\n") for word in dict_one]
    with open(dic_two_path, 'w') as writer:
        [writer.write(word + "\n") for word in dict_two]

    #TODO:corpora_combine_ID

if __name__ == '__main__':
    print('translation data prepare...')
