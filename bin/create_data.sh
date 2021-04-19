#!/usr/bin/env bash

MAX_LEN=500
MIN_COUNT=5
vocab_size=50000

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.pubtator ]; then
    echo "create alignment between CTD and PubTator Central"
    python src/build_data/build_ctd_data.py
fi

# if [ ! -f ${BIORE_DATA_ROOT}/alignment_CTD_PTC.tokenized.pubtator ]; then
#     echo "tokenize abstracts, and prune abstracts longer than MAX_LEN"
#     python src/build_data/learn_bpe.py -i <(cat ${BIORE_DATA_ROOT}/alignment_CTD_PTC.pubtator | grep -e '|t|' -e '|a|' | sed "s/^[0-9]*?|[ta]|//") -o ${BIORE_DATA_ROOT}/vocab_BPE_${vocab_size}.txt -s ${vocab_size}
#     python src/build_data/tokenize.py ${BIORE_DATA_ROOT}/alignment_CTD_PTC.pubtator ${BIORE_DATA_ROOT}/alignment_CTD_PTC.tokenized.pubtator $MAX_LEN
# fi

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.merge_rel.pubtator ]; then
    echo "prune chemical-gene relation types"
    python src/build_data/merge_ctd_relation_type.py ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.pubtator ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.merge_rel.pubtator 1000
fi

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.NULL.pubtator ]; then
    echo "Download abstracts from Pubtator that does not have gene-chem-disease relation labels in CTD records"
    python src/build_data/generate_NULL_abstract.py
fi


echo "split data into train dev test"
python src/build_data/create_data_split.py

echo "build BPE tokenization with vocab_size"
