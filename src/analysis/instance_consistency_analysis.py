import sys
import json
import numpy as np

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
    for ent in entities[::-1]:  # from end to beginning
        start, end = ent["start"], ent["end"]
        # print(text[start:end])
        text = text[:start] + " " + \
            text[start:end].strip("#").strip("@").strip() + " " + text[end:]

    text = " ".join(text.strip().split())
    # print(text)
    if text not in sent2docids:
        sent2docids[text] = [docid]
    else:
        sent2docids[text].append(docid)

avg_bucket_size = []
for t, bucket in sent2docids.items():
    avg_bucket_size.append(len(bucket))
print(f"number of buckets: {len(avg_bucket_size)}; bucket size: mean: {np.mean(avg_bucket_size)}, median: {np.median(avg_bucket_size)}, max: {np.max(avg_bucket_size)}")

bucket_more_than_10 = 0
bucket_more_than_5 = 0
bucket_more_than_2 = 0
bucket_more_than_1 = 0

instance_more_than_10 = 0
instance_more_than_5 = 0
instance_more_than_2 = 0
instance_more_than_1 = 0

for t, bucket in sent2docids.items():
    if len(bucket) > 10:
        bucket_more_than_10 += 1
        instance_more_than_10 += len(bucket)
    if len(bucket) > 5:
        bucket_more_than_5 += 1
        instance_more_than_5 += len(bucket)
    if len(bucket) > 2:
        bucket_more_than_2 += 1
        instance_more_than_2 += len(bucket)
    if len(bucket) > 1:
        bucket_more_than_1 += 1
        instance_more_than_1 += len(bucket)
print(bucket_more_than_10, bucket_more_than_5,
      bucket_more_than_2, bucket_more_than_1)
print(instance_more_than_10, instance_more_than_5,
      instance_more_than_2, instance_more_than_1)


docid2bucketid = {}
for i, (t, bucket) in enumerate(sent2docids.items()):
    for docid in bucket:
        docid2bucketid[docid] = i

with open(pred_path + "/test.results.json", 'r') as f:
    data = json.loads(f.read())

predictions = data["predictions"]


pred_buckets = [[] for _ in range(len(sent2docids))]
label_buckets = [[] for _ in range(len(sent2docids))]
for sent in predictions:
    docid = sent["docid"]
    label = sent["label_names"][0] if len(sent["label_names"]) > 0 else "NA"
    pred = sent["predictions"][0] if len(sent["predictions"]) > 0 else "NA"
    bucketid = docid2bucketid[docid]
    if label != "NA":
        label_buckets[bucketid].append(label)
    if pred != "NA":
        pred_buckets[bucketid].append(pred)

avg_label_size = np.mean([len(set(b)) for b in label_buckets])
avg_pred_size = np.mean([len(set(b)) for b in pred_buckets])
print("avg number of label types per sentence", avg_label_size, avg_pred_size)

variance_bucket_label = []
for label_list in label_buckets:
    label_dict = {}
    for l in label_list:
        if l not in label_dict:
            label_dict[l] = 0
        label_dict[l] += 1
    if len(label_dict) == 0:
        variance_bucket_label.append(0)
    else:
        variance_bucket_label.append(np.mean(list(label_dict.values())))


variance_bucket_pred = []
for label_list in pred_buckets:
    label_dict = {}
    for l in label_list:
        if l not in label_dict:
            label_dict[l] = 0
        label_dict[l] += 1
    if len(label_dict) == 0:
        variance_bucket_pred.append(0)
    else:
        variance_bucket_pred.append(np.mean(list(label_dict.values())))
print("avg number of times each label occurred per sentence",
      np.mean(variance_bucket_label), np.mean(variance_bucket_pred))
