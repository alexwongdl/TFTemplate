"""
Created by Alex Wang
On 2017-09-19
"""
import collections


def _read_words(filename):
    """
    :param filename:
    :return: 所有句子连成一个字符串，切分单词构成一个列表
    """
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    """
    :param filename:
    :return:
        word_to_id--单词到id的映射
        id_to_word--id到单词的映射
    """
    str_list = _read_words(filename)
    counter_one = collections.Counter(str_list)
    count_pairs = sorted(counter_one.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    print("size of word_to_id:" + str(len(word_to_id)))  # 10000
    return word_to_id, id_to_word


if __name__ == "__main__":
    print("__main__")
    word_to_id, id_to_word = build_vocab("../data/ptb/ptb.train.txt")
    for i in range(30):
        print(str(i) + "\t" + id_to_word[i])
    size = len(word_to_id)
    word_to_id.update({"</s>": size}) ## nmt包含<unk>、<eos>、</s>
    id_to_word.update({size :"</s>"})
    print("size of word_to_id:" + str(len(word_to_id)))