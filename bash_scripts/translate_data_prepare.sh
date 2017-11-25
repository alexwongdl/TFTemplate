#!/bin/bash

root_data_path=/home/recsys/hzwangjian1/learntf/TFTemplate/data/translate
python MainEntryNMT_vanilla.py \
    --task=corpus_batch_prepare \    #prepare_data  corpus_to_id corpus_batch_prepare
    --corpora_one_path=${root_data_path}/giga-fren.release2.fixed.fr \
    --corpora_two_path=${root_data_path}/giga-fren.release2.fixed.en \
    --corpora_combine_path=${root_data_path}/giga-fr-en \
    --dict_one_path=${root_data_path}/fr_dict \
    --dict_two_path=${root_data_path}/en_dict \
    --corpora_combine_ID_path=${root_data_path}/giga-fr-en-id \
    --batch_size=16 \
    --corpora_raw_id_dir=${root_data_path}/subfiles \
    --corpora_format_dir=/home/recsys/hzwangjian1/tensorflow/nmt/subfiles_format

