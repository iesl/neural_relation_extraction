import os, sys, json
from tqdm import tqdm
data_path = os.environ["BIORE_DATA_ROOT"]
#input_file = data_path + "/CTD_all_entities_pubtator_interactions"
input_file = sys.argv[1]
output_file = sys.argv[2]
freq_threshold = int(sys.argv[3])
# collect chemical-gene relation hierarchy
relation_info = {}
relation_hierarchy_bottom_up = {}
for line in open(data_path + "/new_ctd/origin/CTD_chem_gene_ixn_types.tsv"):
    if line[0] != "#":
        l = line.split("\t")
        name, code, description, parent_code = l[:4]
        name = "_".join(name.split())
        parent_code = parent_code.strip()
        relation_info[name] = (name, code, description)
        relation_info[code] = (name, code, description)
        if parent_code == "":
            relation_hierarchy_bottom_up[code] = None
        else:
            relation_hierarchy_bottom_up[code] = parent_code


def find_ancester(rel):
    code = relation_info[rel][1]
    parent_code = relation_hierarchy_bottom_up[code]
    while parent_code != None:
        code = parent_code
        parent_code = relation_hierarchy_bottom_up[code]
    return relation_info[code][0]

def print_hierarchy(data):
    for k, v in data.items():
        dfs_list = [relation_info[k][0]]
        while v != None:
            dfs_list = [relation_info[v][0]] + dfs_list
            v = data[v]
        print('\t'.join(dfs_list))

print_hierarchy(relation_hierarchy_bottom_up) 

# count frequency of each relation and decide whether to use affect or to use increase/decrease and drop affect
chem_gene_relation_freq = {}
#for line in open(data_path + "/alignment_CTD_PTC.pubtator"):
for line in open(input_file):
    l = line.split("\t")
    if len(l) != 5 or l[1].split(":")[0] != "chem_gene":
        pass
    else:
        corr, rel = l[1].split(":")[1].split("^")
        rel = "_".join(rel.split())
        rel = find_ancester(rel)
        rel = corr + "^" + rel
        if rel not in chem_gene_relation_freq:
            chem_gene_relation_freq[rel] = 0
        chem_gene_relation_freq[rel] += 1

# decide whether to use affects or use increases/decreases
# Currently this dictionary is a placeholder, all "increases"/"decreases" collapse to "affects"
relation_common = set()
for rel, freq in list(chem_gene_relation_freq.items()):
    if freq < freq_threshold:
        print(f"{rel} removed")
        continue
    relation_common.add(rel)
    # if increase and decrease occurs more than 1000 times, and each occurs more often than affects
    #if freqs[1] > 1000 and freqs[2] > 1000 and freqs[0] < freqs[1] and freqs[0] < freqs[2]:
    # if freqs[0] < freqs[1] and  freqs[0] < freqs[2]:
    #     relation_common[rel] = True
    # else:
    #     relation_common[rel] = True
    

# output with collaped relation type
fout = open(output_file,"w")
article = ""
relations = {}
count_articles_before = 0
count_articles_after = 0
relation_freq = {}
for line in open(input_file):
    #print(article)
    l = line.strip("\n").split("\t")
    if line.strip() == "": # when meet a blank line, output the previous article
        if len(relations) > 0: # if relation is blank, skip previous article
            relations_ = ""
            for k, rels in relations.items():
                for rel, desc in rels:
                    relations_ += "\t".join([k[0], rel, k[1], k[2], desc]) + "\n"
            fout.write(article + relations_ + "\n")
            count_articles_after += 1
        count_articles_before += 1
        article, relations = "", {}
    elif len(l) != 5:
        article += line
    else:
        if l[1].split(":")[0] != "chem_gene":
            rel = l[1]
            if (l[0], l[2], l[3]) not in relations:
                relations[(l[0], l[2], l[3])] = set()
            relations[(l[0], l[2], l[3])].add((rel, l[4]))
            if rel not in relation_freq: relation_freq[rel] = 0
            relation_freq[rel] += 1
        else:
            corr, rel = l[1].split(":")[1].split("^")
            rel = "_".join(rel.split())
            rel = find_ancester(rel)
            rel = corr + "^" + rel
            # filter low frequency relation types
            if rel not in relation_common:
                continue
            
            if (l[0], l[2], l[3]) not in relations: # docid, e1, e2
                relations[(l[0], l[2], l[3])] = set()
            rel = "chem_gene:" + rel
            if rel not in relations[(l[0], l[2], l[3])]:
                relations[(l[0], l[2], l[3])].add((rel, l[4]))
            
            if rel not in relation_freq: relation_freq[rel] = 0
            relation_freq[rel] += 1
            
            
fout.close()
print(f"{count_articles_after}/{count_articles_before} articles left")


sorted_relation_types = sorted(list(relation_freq.items()), key=lambda x:x[1], reverse=True)
print(sorted_relation_types)
relation_map = [(r, i) for i, (r, f) in enumerate(sorted_relation_types)]
relation_map = dict(relation_map)
open(data_path + "/new_ctd/relation_map.json","w").write(json.dumps(relation_map, indent="\t"))

#fout = open(data_path + "/new_ctd/relation_descriptions.txt","w")
#fout.write("relation\tdescription\n")
#for name, info in relation_info.items():
#    if name == info[0] and find_ancester(name) == name and find_ancester(name) in relation_common:
#        fout.write("chem_gene:%s\t%s\n" %(find_ancester(name), info[2]))
#fout.close()
