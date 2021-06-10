import sys
import json
import numpy as np
from scipy import stats

data_path = sys.argv[1]  # json file that contains original data
pred_path = sys.argv[2]  # json file that contains predictions and labels


with open(data_path + "/test.json") as f:
    data = json.loads(f.read())
    print(len(data))

sent2docids = {}
for sent in data:
    docid = sent["docid"]
    text = sent["text"]
    #print(docid, text)
    entities = sent["entity"]
    entities = sorted(entities, key=lambda x: x["start"])
    offsets = []
    for ent in entities[::-1]:  # from end to beginning
        start, end, role = ent["start"], ent["end"], ent["id"]
        offsets.append([start, end, role])
        # print(text[start:end])
        text = text[:start] + " " + \
            text[start:end].strip("#").strip("@").strip() + " " + text[end:]

    text = " ".join(text.strip().split())
    # print(text)
    if text not in sent2docids:
        sent2docids[text] = [(docid, offsets)]
    else:
        sent2docids[text].append((docid, offsets))

docid2bucket = {}
for i, (text, bucket) in enumerate(sent2docids.items()):
    for docid, offsets in bucket:
        docid2bucket[docid] = bucket


with open(pred_path + "/test.results.json", 'r') as f:
    data = json.loads(f.read())

predictions = data["predictions"]


docid_pred = {}
docid_label = {}
for sent in predictions:
    docid = sent["docid"]
    label = sent["label_names"][0] if len(sent["label_names"]) > 0 else "NA"
    pred = sent["predictions"][0] if len(sent["predictions"]) > 0 else "NA"
    docid_pred[docid] = pred
    docid_label[docid] = label


def calculate_dist(offsets1, offsets2):
    dist_list = []
    for start1, end1, role1 in offsets1:
        dist = 10000
        for start2, end2, role2 in offsets2:
            dist_ = min(abs(start1 - start2), abs(start1 - end2),
                        abs(end1 - start2), abs(end1 - end2))
            dist = min(dist_, dist)
        dist_list.append(dist)
    return np.mean(dist_list)

# def calculate_dist(offsets1, offsets2):
#     dist_list = []
#     for start1, end1, role1 in offsets1:
#         dist = 10000
#         for start2, end2, role2 in offsets2:
#             if role1 == role2:
#                 dist_ = min(abs(start1 - start2), abs(start1 - end2),
#                             abs(end1 - start2), abs(end1 - end2))
#                 dist = min(dist_, dist)
#         dist_list.append(dist)
#     return np.mean(dist_list)


neg_dist_list = []
pred_list = []
for docid in docid_pred:
    buckets = docid2bucket[docid]
    offsets = None
    for docid_, offsets_ in buckets:
        if docid_ == docid:
            offsets = offsets_
    for docid_, offsets_ in buckets:
        if docid_ != docid:
            if docid_pred[docid_] == "NA" or docid_pred[docid] == "NA":
                continue
            neg_dist_list.append(-calculate_dist(offsets, offsets_))
            if docid_pred[docid_] == docid_pred[docid]:
                pred_list.append(1)
            else:
                pred_list.append(0)

spearman_corr_pred = stats.spearmanr(neg_dist_list, pred_list)
print(spearman_corr_pred)


neg_dist_list = []
label_list = []
for docid in docid_label:
    buckets = docid2bucket[docid]
    offsets = None
    for docid_, offsets_ in buckets:
        if docid_ == docid:
            offsets = offsets_
    for docid_, offsets_ in buckets:
        if docid_ != docid:
            if docid_label[docid_] == "NA" or docid_label[docid] == "NA":
                continue
            neg_dist_list.append(-calculate_dist(offsets, offsets_))
            if docid_label[docid_] == docid_label[docid]:
                label_list.append(1)
            else:
                label_list.append(0)

spearman_corr_label = stats.spearmanr(neg_dist_list, label_list)
print(spearman_corr_label)
