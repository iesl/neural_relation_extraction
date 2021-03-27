import numpy as np
import torch
import os, sys
import json
from tqdm import tqdm
from sklearn import metrics

__all__ = [
    "Trainer",
]

class Trainer(object):

    def __init__(self, data=None, model=None, logger=None, config=None, device=None,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        
        self.data = data
        self.model = model
        self.logger = logger
        self.config = config
        self.device = device
        # setup optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=config["learning_rate"], weight_decay=0.0
        )
        if self.config["output_path"] == "":
            base_dir_name = "_".join([self.config["encoder_type"].split("/")[-1],
                                        self.config["aggregation"],
                                        str(self.config["train_batch_size"]),
                                        str(self.config["test_batch_size"]),
                                        str(self.config["max_text_length"]),
                                        str(self.config["dim"]),
                                        str(self.config["learning_rate"]),
                                        str(self.config["epochs"]),
                                        str(self.config["patience"]),
                                        #str(self.config["log_interval"]),
                                        str(self.config["seed"])])
            self.output_path = os.path.join(os.environ['BIORE_ROOT'], "saved_models", 
                                            os.path.basename(self.config["data_path"]), 
                                            base_dir_name)
        else:
            self.output_path = self.config["output_path"]
        #for name, param in self.model.named_parameters():
        #    #if param.requires_grad:
        #    print(name)


    def save_model(self, best_metric_threshold):
        model_state_dict = self.model.state_dict() # TODO may have device issue
        checkpoint = {
            'model': model_state_dict,
            'opt': self.opt,
            'threshold': best_metric_threshold
        }
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        checkpoint_path = os.path.join(self.output_path, 'model.pt')
        self.logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)


    def load_model(self):
        checkpoint_path = os.path.join(self.output_path, 'model.pt')
        self.logger.info("Loading best checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Model {checkpoint_path} does not exist.")
            return 0
        checkpoint = torch.load(checkpoint_path)
        self.opt = vars(checkpoint['opt'])
        self.model.load_state_dict(checkpoint['model'])
        return checkpoint['threshold']
        

    def train(self):
        self.logger.debug("This is training")
        #inputs = tokenizer("Hello world!", return_tensors="pt")
        #outputs = self.model(**inputs)
        bcelogitloss = torch.nn.BCEWithLogitsLoss()
        best_metric = 0
        best_metric_threshold = {}
        patience = 0
        rolling_loss = []
        max_step = self.config["epochs"] * len(self.data)
        
        for i, (input_ids, token_type_ids, attention_mask, ep_mask, label_array) in iter(self.data):
            #self.logger.debug(f"training {i}")
            self.opt.zero_grad()

            """Loss"""
            #self.logger.debug(input_ids.shape, token_type_ids.shape, attention_mask.shape, ep_mask.shape)
            scores = self.model(input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device), ep_mask.to(self.device)) # (batchsize, R)
            loss = bcelogitloss(scores, label_array.to(self.device))

            """End"""
            loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
            self.opt.step()

            rolling_loss.append(float(loss.detach().cpu()))
            #self.logger.debug(f"training {loss.detach().cpu()}")
            # print training loss 
            if i % 100 < self.config["train_batch_size"]:
                self.logger.info(f"{i}-th {i%100} loss: {np.mean(rolling_loss)}")
                rolling_loss = []

            # evaluate on dev set
            if i % self.config["log_interval"] < self.config["train_batch_size"]:
                macro_perf, micro_perf, per_rel_perf = self.test(test=False) # per_rel_perf is a list of length of (number of relation types + 1)
                
                if macro_perf["F"] > best_metric:
                    best_metric = macro_perf["F"]
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt}")
                        best_metric_threshold[rel_name] = tt
                    patience = 0
                    self.save_model(best_metric_threshold)
                    
                else:
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV: Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV: Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        self.logger.info(f"{i * 100 / len(self.data)} % DEV: {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt}")
                    patience += 1
            
            # early stop
            if patience > self.config["patience"]:
                self.logger.info("triggers early stop; ended")
                break
            if i > max_step:
                self.logger.info("exceeds maximum steps; ended")
                break
        # load best model
        best_metric_threshold = self.load_model()
        macro_perf, micro_perf, per_rel_perf = self.test(test=True, best_metric_threshold=best_metric_threshold)
        self.logger.info(f"Test: Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
        self.logger.info(f"Test: Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
        for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
            self.logger.info(f"TEST: {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt}")

    

    def test(self, test=False, best_metric_threshold=None):
        if test==True:
            self.logger.debug("This is testing")
            # if in test mode, use existing metric thresholds
            assert best_metric_threshold!=None, "evaluation on test data requires best_metric_threshold"
            # output results to a seperate file
            fout = open(os.path.join(self.output_path, "test.results.json"), "w")
            fout_json = {"threshold": {}, "predictions":[], "results":{"macro":{}, "micro":{}, "per_rel":{}}}

            data = self.data.test
            threshold_vec = np.zeros(len(self.data.relation_map))
            for rel, thres in best_metric_threshold.items():
                relid = self.data.relation_map[rel]
                threshold_vec[relid] = thres
                fout_json["threshold"][rel] = float(thres)

        else:
            self.logger.debug("This is validation")
            # if in validation mode, tune thresholds for best f1 per relation
            data = self.data.val
            threshold_vec = np.zeros(len(self.data.relation_map))
        
        # getting scores for all valid/test data
        with torch.no_grad():
            scores = np.zeros((len(data), len(self.data.relation_map))) # (num_test, R)
            labels = np.zeros((len(data), len(self.data.relation_map))) # (num_test, R)
            self.logger.debug(f"length of data: {len(data)}")
            input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id = [], [], [], [], [], [], [], []

            for i, dict_ in tqdm(enumerate(data)):
                input_array.append(dict_["input"])
                pad_array.append(dict_["pad"])
                label_array.append(dict_["label_vector"])
                e1_indicators.append(dict_["e1_indicators"])
                e2_indicators.append(dict_["e2_indicators"])
                docid.append(dict_["docid"])
                e1id.append(dict_["e1"])
                e2id.append(dict_["e2"])

                if len(input_array) == self.config["test_batch_size"] or i == len(data) - 1:

                    input_ids = torch.tensor(np.array(input_array), dtype=torch.long).to(self.device)
                    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long).to(self.device)
                    attention_mask = torch.tensor(np.array(pad_array), dtype=torch.long).to(self.device)
                    ep_mask = np.full((len(input_array), self.data.max_text_length, self.data.max_text_length), -1e8)
                    for j in range(len(input_array)):
                        ep_outer = 1 - np.outer(e1_indicators[j], e2_indicators[j])
                        ep_mask[j] = ep_mask[j] * ep_outer
                    ep_mask = torch.tensor(ep_mask, dtype=torch.float).to(self.device)
                    
                    score = self.model(input_ids, token_type_ids, attention_mask, ep_mask).detach().cpu().numpy() # (b, R)
                    #print(input_ids.shape, ep_mask.shape, score.shape, i+1-len(input_array), i+1, len(labels))
                    sys.stdout.flush()

                    scores[(i+1-len(input_array)):(i+1), :] = score
                    labels[(i+1-len(input_array)):(i+1), :] = np.array(label_array)

                    if test == True:
                        # in test mode, save predictions for each data point (docid, e1, e2)
                        prediction = (score > threshold_vec) # (b, R)
                        for j in range(len(input_array)):
                            predict_names = []
                            for k in list(np.where(prediction[j] == 1)[0]):
                                predict_names.append(self.data.relation_name[k])
                            label_names = []
                            for k in list(np.where(label_array[j] == 1)[0]):
                                label_names.append(self.data.relation_name[k])
                            score_dict = {}
                            for k, scr in enumerate(list(score[j])):
                                score_dict[self.data.relation_name[k]] = float(scr)
                            fout_json["predictions"].append({"docid": docid[j], "e1": e1id[j], "e2": e2id[j], "label_names": label_names, "predictions": predict_names, "scores": score_dict})
                    
                    input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id = [], [], [], [], [], [], [], []

        def calculate_metrics(predictions, labels):
            TP = predictions * labels
            #FP = predictions * (labels == 0.0)
            #FN = (predictions == 0.0) * labels

            if predictions.sum() == 0:
                micro_p = 0.0
            else:
                micro_p = TP.sum() / predictions.sum()
            if labels.sum() == 0:
                micro_r = 0.0
            else:
                micro_r = TP.sum() / labels.sum()
            if micro_p == 0.0 or micro_r == 0.0:
                micro_f = 0.0
            else:
                micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)

            per_rel_p = np.zeros(predictions.shape[1])
            per_rel_r = np.zeros(predictions.shape[1])
            per_rel_f = np.zeros(predictions.shape[1])
            # per relation metrics:
            for i in range(predictions.shape[1]):
                TP_ = TP[:, i].sum()
                P_ = predictions[:, i].sum()
                T_ = labels[:, i].sum()

                if T_ == 0: # if no such relation in the test data, recall = 0
                    per_rel_r[i] = 0.0
                else:
                    per_rel_r[i] = TP_ / T_
                
                if P_ == 0: # if no such relation in the prediction, precision = 0
                    per_rel_p[i] = 0.0
                else:
                    per_rel_p[i] = TP_ / P_

                if per_rel_p[i] == 0.0 or per_rel_r[i] == 0.0:
                    per_rel_f[i] = 0.0
                else:
                    per_rel_f[i] = 2 * per_rel_p[i] * per_rel_r[i] / (per_rel_p[i] + per_rel_r[i])

            
            macro_p = per_rel_p.mean()
            macro_r = per_rel_r.mean()
            macro_f = per_rel_f.mean()
            return micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f

        if test==True:
            # in test mode, use existing thresholds, save results.
            predictions = (scores > threshold_vec) # (num_test, R)
            micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f = calculate_metrics(predictions, labels)
            
            fout_json["results"]["micro"] = {"P": float(micro_p), "R": float(micro_r), "F": float(micro_f)}
            fout_json["results"]["macro"] = {"P": float(macro_p), "R": float(macro_r), "F": float(macro_f)}
            for i, rel_name in self.data.relation_name.items():
                fout_json["results"]["per_rel"][rel_name] = {"P": float(per_rel_p[i]), "R": float(per_rel_r[i]), "F": float(per_rel_f[i])}
            #print(fout_json)
            fout.write(json.dumps(fout_json, indent="\t"))
            fout.close()

        else:
            # in validation model, tune the thresholds
            for i, rel_name in self.data.relation_name.items():
                prec_array, recall_array, threshold_array = metrics.precision_recall_curve(labels[:, i], scores[:, i])
                f1_array = 2 * prec_array * recall_array / (prec_array + recall_array)
                best_threshold = threshold_array[np.argmax(f1_array)]
                threshold_vec[i] = best_threshold

            predictions = (scores > threshold_vec) # (num_test, R)
            micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f = calculate_metrics(predictions, labels)


        metric = {}
        macro_perf = {"P": macro_p, "R": macro_r, "F": macro_f}
        micro_perf = {"P": micro_p, "R": micro_r, "F": micro_f}
        per_rel_perf = {}
        for i, rel_name in self.data.relation_name.items():
            per_rel_perf[rel_name] = [per_rel_p[i], per_rel_r[i], per_rel_f[i], threshold_vec[i]]
        return macro_perf, micro_perf, per_rel_perf
