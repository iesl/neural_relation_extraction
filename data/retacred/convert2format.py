import json
import sys

marker_set = set([])
def read_tacred_format(input_path):
    input_data = json.loads(open(input_path).read())
    output_data = []
    for inp in input_data:
        d = {}
        d["docid"] = inp["id"]
        e1_start, e1_end, e1_type = int(inp["subj_start"]), int(inp["subj_end"]) + 1, inp["subj_type"]
        e2_start, e2_end, e2_type = int(inp["obj_start"]), int(inp["obj_end"]) + 1, inp["obj_type"]
        if e1_start < e2_start:
             
            tokens_with_marker = inp["token"][:e1_start] + \
                                 ["@"] + \
                                 inp["token"][e1_start:e1_end] + \
                                 ["@"] + \
                                 inp["token"][e1_end:e2_start] + \
                                 ["#"] + \
                                 inp["token"][e2_start:e2_end] + \
                                 ["#"] + \
                                 inp["token"][e2_end:]
            e1_start = e1_start
            e1_end = e1_end + 2
            e2_start = e2_start + 2
            e2_end = e2_end + 4
        else:
            tokens_with_marker = inp["token"][:e2_start] + \
                                 ["#"] + \
                                 inp["token"][e2_start:e2_end] + \
                                 ["#"] + \
                                 inp["token"][e2_end:e1_start] + \
                                 ["@"] + \
                                 inp["token"][e1_start:e1_end] + \
                                 ["@"] + \
                                 inp["token"][e1_end:]
            e2_start = e2_start
            e2_end = e2_end + 2
            e1_start = e1_start + 2
            e1_end = e1_end + 4
        if tokens_with_marker[0] in ["@", "#"]:
            tokens_with_marker = [""] + tokens_with_marker
            e1_start += 1
            e1_end += 1
            e2_start += 1
            e2_end += 1

        marker_set.add("@")
        marker_set.add("#")

        tokenpos2charpos = [0]
        p = 0
        d["text"] = ""
        for i, t in enumerate(list(tokens_with_marker)):
            p += len(t) + 1
            d["text"] += t + " "
            tokenpos2charpos.append(p)

        e1sp, e1ep = tokenpos2charpos[int(e1_start)], tokenpos2charpos[int(e1_end)] - 1
        e2sp, e2ep = tokenpos2charpos[int(e2_start)], tokenpos2charpos[int(e2_end)] - 1
        e1_info = {"start": e1sp, "end": e1ep, "mention": d["text"][e1sp:e1ep], "type": e1_type, "id": "e1"}
        e2_info = {"start": e2sp, "end": e2ep, "mention": d["text"][e2sp:e2ep], "type": e2_type, "id": "e2"}
        d["entity"] = [e1_info, e2_info]

        rel_type = inp["relation"]
        d["relation"] = [{"type":rel_type, "subj": "e1", "obj": "e2"}]
        output_data.append(d)
    return output_data

data_list = read_tacred_format(sys.argv[1] + "/train.json")
open("train.json", "w").write(json.dumps(data_list, indent="\t"))
data_list = read_tacred_format(sys.argv[1] + "/dev.json")
open("valid.json", "w").write(json.dumps(data_list, indent="\t"))
data_list = read_tacred_format(sys.argv[1] + "/test.json")
open("test.json", "w").write(json.dumps(data_list, indent="\t"))
open("entity_type_markers.json","w").write(json.dumps(list(marker_set), indent="\t"))
