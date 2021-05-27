import sys
import json
import numpy as np
import matplotlib.pyplot as plt


input_json = sys.argv[1]


data = json.loads(open(input_json).read())
rel_type = list(data["threshold"].keys())
tags = []
rel2id = {}
for i, r in enumerate(rel_type):
    rel2id[r] = i
    tags.append(r)
rel2id["NA"] = len(rel2id)
tags.append("NA")


confusion_mat = np.zeros((len(rel2id), len(rel2id)))

for p in data["predictions"]:
    if len(p["label_names"]) == 0:
        gt = "NA"
    else:
        gt = p["label_names"][0]
    if len(p["predictions"]) == 0:
        pd = "NA"
    else:
        pd = p["predictions"][0]
    row, col = rel2id[gt], rel2id[pd]
    confusion_mat[row, col] += 1

confusion_mat = np.log(confusion_mat+1)
fig, ax = plt.subplots()
im = ax.imshow(confusion_mat)
#'''
ax.set_xticks(np.arange(len(tags)))
ax.set_yticks(np.arange(len(tags)))
ax.set_xticklabels(tags, fontdict={"fontsize": 3})
ax.set_yticklabels(tags, fontdict={"fontsize": 3})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(tags)):
    for j in range(len(tags)):
        text = ax.text(j, i, '{0:.2f}'.format(confusion_mat[i, j]),
                       ha="center", va="center", color="w", fontsize=2)
#'''
ax.set_title("Confusion matrix")
fig.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
