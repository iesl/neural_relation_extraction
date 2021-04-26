import os, sys
from tqdm import tqdm
import random
import json
random.seed(1234)

data_path = os.environ["BIORE_DATA_ROOT"]

# proportions below are over document level.
proportion_null = 0.1 
val_split = ['2018']
test_split = ['2019', '2020']

date_map = {}
for line in open(data_path + "/new_ctd/pid2year.map"):
    l = line.strip().split()
    date_map[l[0]] = l[1]

data_list = []
data = ""
for line in open(data_path + "/new_ctd/alignment_CTD_PTC.merge_rel.pubtator"):
    if line.strip() == "" and data != "":
        data_list.append(data)
        data = ""
        continue
    data += line

max_num_null = len(data_list) * (proportion_null / (1 - proportion_null))
count_null = 0
for line in open(data_path + "/new_ctd/alignment_CTD_PTC.NULL.pubtator"):
    if line.strip() == "" and data != "":
        data_list.append(data)
        data = ""
        count_null += 1
        if count_null > max_num_null:
            break
        else:
            continue
    data += line

random.shuffle(data_list)

def output_split(data_list, name):
    
    fout_stat = open(f"{data_path}/new_ctd/{name}_stats.txt","w")
    stats = {}
    with open(f"{data_path}/new_ctd/{name}.json","w") as fout:
        data_json_list = []
        for data in data_list:
            data_json = {"entity":[], "relation":[], "title":None, "abstract":None, "docid":None}
            for line in data.strip("\n").split("\n"):
                l = line.split("\t")
                if len(l) == 5:
                    data_json["relation"].append({"type": l[1], "subj": l[2], "obj": l[3], "desc": l[4]})
                    rel = l[1]
                elif len(l) == 6:
                    data_json["entity"].append({"start":int(l[1]), "end":int(l[2]), "mention":l[3], "type":l[4], "id":l[5]})
                elif "|t|" in line:
                    data_json["title"] = line.split("|t|")[1] + " "
                    data_json["docid"] = line.split("|t|")[0]
                elif "|a|" in line:
                    data_json["abstract"] = line.split("|a|")[1]
            if name == 'valid' and date_map[data_json["docid"]] not in val_split:
                continue
            elif name == 'test' and date_map[data_json["docid"]] not in test_split:
                continue
            elif name == 'train' and date_map[data_json["docid"]] in (val_split + test_split):
                continue
            else:
                pass


            for rel_info in data_json["relation"]:
                rel = rel_info["type"]
                if rel not in stats: stats[rel] = 0
                stats[rel] += 1
            
            data_json_list.append(data_json)
        fout.write(json.dumps(data_json_list, indent="\t"))
    for k, v in stats.items():
        fout_stat.write(f"{k}\t{v}\n")
    fout_stat.close()

output_split(data_list, "train")
output_split(data_list, "valid")
output_split(data_list, "test")
