#!/bin/bash

root_data_path=/home/recsys/hzwangjian1/learntf/TFTemplate/data/translate
python MainEntryNMT_vanilla.py \
    --task=prepare_data \
    --corpora_one_path=${root_data_path}/giga-fren.release2.fixed.fr \
    --corpora_two_path=${root_data_path}/giga-fren.release2.fixed.en \
    --corpora_combine_path=${root_data_path}/giga-fr-en \
    --dict_one_path=${root_data_path}/fr_dict \
    --dic_two_path=${root_data_path}/en_dict \
    --corpora_combine_ID_path=${root_data_path}/giga-fr-en-id
