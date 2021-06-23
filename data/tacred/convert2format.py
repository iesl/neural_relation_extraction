import json
import sys
import random
random.seed(1234)


number_perturb = 10
marker_set = set([])
def read_tacred_format(input_path, perturb=True):
    input_data = json.loads(open(input_path).read())
    output_data = []
    for inp in input_data:
        d = {}
        d["docid"] = inp["id"]
        e1_start, e1_end, e1_type = int(inp["subj_start"]), int(inp["subj_end"]) + 1, inp["subj_type"]
        e2_start, e2_end, e2_type = int(inp["obj_start"]), int(inp["obj_end"]) + 1, inp["obj_type"]

        tokens = [""] + inp["token"]
        e1_start += 1
        e1_end += 1
        e2_start += 1
        e2_end += 1

        if perturb == True:
            num_tokens = len(tokens)
            candidate_list = []
            for ind in range(1, num_tokens):
                if (ind >= e1_start and ind < e1_end) or (ind >= e2_start and ind < e2_end):
                    continue
                elif tokens[ind] in tokens[e1_start:e1_end] or tokens[ind] in tokens[e2_start:e2_end]:
                    continue
                else:
                    candidate_list.append(ind)
            perturb_index = []
            if len(candidate_list) == 0:
                print(tokens, tokens[e1_start:e1_end], tokens[e2_start:e2_end])
            else:
                for _ in range(number_perturb):
                    ind = random.choice(candidate_list)
                    if random.random() > 0.5:
                        perturb_index.append([ind, ind+1, e2_start, e2_end])
                    else:
                        perturb_index.append([e1_start, e1_end, ind, ind+1])


        tokenpos2charpos = [0]
        p = 0
        for i, t in enumerate(list(tokens)):
            p += len(t) + 1
            tokenpos2charpos.append(p)

        # original data
        if e1_start < e2_start:
            d["text"] = " ".join(tokens[:e1_start] + ["@"] + tokens[e1_start:e1_end] + ["@"] + tokens[e1_end:e2_start] + ["#"] + tokens[e2_start:e2_end] + ["#"] + tokens[e2_end:])
            e1sp, e1ep = tokenpos2charpos[int(e1_start)], tokenpos2charpos[int(e1_end)] - 1 + 4
            e2sp, e2ep = tokenpos2charpos[int(e2_start)] + 4, tokenpos2charpos[int(e2_end)] - 1 + 8
        else:
            d["text"] = " ".join(tokens[:e2_start] + ["#"] + tokens[e2_start:e2_end] + ["#"] + tokens[e2_end:e1_start] + ["@"] + tokens[e1_start:e1_end] + ["@"] + tokens[e1_end:])
            e1sp, e1ep = tokenpos2charpos[int(e1_start)] + 4, tokenpos2charpos[int(e1_end)] - 1 + 8
            e2sp, e2ep = tokenpos2charpos[int(e2_start)], tokenpos2charpos[int(e2_end)] - 1 + 4
        marker_set.add("@")
        marker_set.add("#")
            
        e1_info = {"start": e1sp, "end": e1ep, "mention": d["text"][e1sp:e1ep], "type": e1_type, "id": "e1"}
        e2_info = {"start": e2sp, "end": e2ep, "mention": d["text"][e2sp:e2ep], "type": e2_type, "id": "e2"}
        d["entity"] = [e1_info, e2_info]

        rel_type = inp["relation"]
        d["relation"] = [{"type":rel_type, "subj": "e1", "obj": "e2"}]
        
        # perturbation data
        if perturb == True:
            d["perturbation"] = []
            for e1_start_, e1_end_, e2_start_, e2_end_ in perturb_index:
                d_ = {}
                if e1_start_ < e2_start_:
                    d_["text"] = " ".join(tokens[:e1_start_] + ["@"] + tokens[e1_start_:e1_end_] + ["@"] + tokens[e1_end_:e2_start_] + ["#"] + tokens[e2_start_:e2_end_] + ["#"] + tokens[e2_end_:])
                    e1sp_, e1ep_ = tokenpos2charpos[int(e1_start_)], tokenpos2charpos[int(e1_end_)] - 1 + 4
                    e2sp_, e2ep_ = tokenpos2charpos[int(e2_start_)] + 4, tokenpos2charpos[int(e2_end_)] - 1 + 8
                else:
                    d_["text"] = " ".join(tokens[:e2_start_] + ["#"] + tokens[e2_start_:e2_end_] + ["#"] + tokens[e2_end_:e1_start_] + ["@"] + tokens[e1_start_:e1_end_] + ["@"] + tokens[e1_end_:])
                    e1sp_, e1ep_ = tokenpos2charpos[int(e1_start_)] + 4, tokenpos2charpos[int(e1_end_)] - 1 + 8
                    e2sp_, e2ep_ = tokenpos2charpos[int(e2_start_)], tokenpos2charpos[int(e2_end_)] - 1 + 4
                e1_info_ = {"start": e1sp_, "end": e1ep_, "mention": d_["text"][e1sp_:e1ep_], "type": e1_type if d_["text"][e1sp_:e1ep_] == d["text"][e1sp:e1ep] else "null"}
                e2_info_ = {"start": e2sp_, "end": e2ep_, "mention": d_["text"][e2sp_:e2ep_], "type": e2_type if d_["text"][e2sp_:e2ep_] == d["text"][e2sp:e2ep] else "null"}
                d_["entity"] = [e1_info_, e2_info_]
                d["perturbation"].append(d_)

        output_data.append(d)
    return output_data

data_list = read_tacred_format(sys.argv[1] + "/train.json", perturb=True)
open("train.json", "w").write(json.dumps(data_list, indent="\t"))
data_list = read_tacred_format(sys.argv[1] + "/dev.json", perturb=False)
open("valid.json", "w").write(json.dumps(data_list, indent="\t"))
data_list = read_tacred_format(sys.argv[1] + "/test.json", perturb=False)
open("test.json", "w").write(json.dumps(data_list, indent="\t"))
open("entity_type_markers.json","w").write(json.dumps(list(marker_set), indent="\t"))
