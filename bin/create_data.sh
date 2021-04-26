#!/usr/bin/env bash

MIN_FREQ=1000

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.pubtator ]; then
    echo "create alignment between CTD and PubTator Central"
    python src/build_data/build_ctd_data.py
fi

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.merge_rel.pubtator ]; then
    echo "prune chemical-gene relation types"
    python src/build_data/merge_ctd_relation_type.py ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.pubtator ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.merge_rel.pubtator $MIN_FREQ
fi

if [ ! -f ${BIORE_DATA_ROOT}/new_ctd/alignment_CTD_PTC.NULL.pubtator ]; then
    echo "Download abstracts from Pubtator that does not have gene-chem-disease relation labels in CTD records"
    python src/build_data/generate_NULL_abstract.py
fi

echo "download pid-to-year map"
python src/build_data/download_pid_date.py

echo "split data into train dev test"
python src/build_data/create_data_split_by_time.py

