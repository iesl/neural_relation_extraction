import numpy as np
import torch
import os, sys
import json
from tqdm import tqdm
import wandb
from sklearn import metrics
from scipy.special import logsumexp
from module.utils import log1mexp

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
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        self.bcelogitloss = torch.nn.BCEWithLogitsLoss() # y is 1 or 0, x is 1-d logit
        self.celogitloss = torch.nn.CrossEntropyLoss() # y is a non-negative integer, x is a multi dimensional logits
        
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
        checkpoint_path = os.path.join(self.config["output_path"], 'model.pt')
        self.logger.info("Saving checkpoint %s" % checkpoint_path)
        torch.save(checkpoint, checkpoint_path)


    def load_model(self):
        checkpoint_path = os.path.join(self.config["output_path"], 'model.pt')
        self.logger.info("Loading best checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Model {checkpoint_path} does not exist.")
            return 0
        checkpoint = torch.load(checkpoint_path)
        self.opt = vars(checkpoint['opt'])
        self.model.load_state_dict(checkpoint['model'])
        return checkpoint['threshold']


    def train(self):
        if self.config["wandb"]:
            wandb.init(project="re", settings=wandb.Settings(start_method="fork"))
            wandb.config.update(self.config, allow_val_change=True)

        self.logger.debug("This is training")
        #inputs = tokenizer("Hello world!", return_tensors="pt")
        #outputs = self.model(**inputs)
        if self.config["multi_label"] == True and self.config["score_func"] != "box":
            loss_func = lambda input, target: self.bcelogitloss(input, target)
        elif self.config["multi_label"] == True and self.config["score_func"] == "box":
            loss_func = lambda input, target: -(target * input + (1 - target) * log1mexp(input)).mean() # bce log loss
        elif self.config["multi_label"] == False and self.config['na_hierarchy'] == False:
            loss_func = lambda input, target: self.celogitloss(input, target) # input are logits 
        elif self.config["multi_label"] == False and self.config['na_hierarchy'] == True:
            loss_func = lambda input, target: -input[torch.arange(target.shape[0]).long(), target.long()].mean()  # input are log prob


        best_metric = 0
        best_metric_threshold = {}
        patience = 0
        rolling_loss = []
        max_step = self.config["epochs"] * len(self.data)
        
        for i, (input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator, label_array) in iter(self.data):
            #self.logger.debug(f"training {i}")
            self.opt.zero_grad()
            

            """Loss"""
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            ep_mask = ep_mask.to(self.device)
            e1_indicator = e1_indicator.to(self.device)
            e2_indicator = e2_indicator.to(self.device)
            #self.logger.debug(input_ids.shape, token_type_ids.shape, attention_mask.shape, ep_mask.shape)
            scores = self.model(input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator) # (batchsize, R) or (batchsize, R+1)
            if self.config["multi_label"] == True:
                loss = loss_func(scores, label_array.to(self.device))
            else: 
                #scores = torch.cat([scores, -scores.logsumexp(1, keepdim=True)], dim=1) # (batchsize, R+1)
                #scores = torch.cat([scores, -scores.max(1, keepdim=True)[0]], dim=1) # (batchsize, R+1)
                label_array = torch.cat([label_array, 1 - (label_array.sum(1, keepdim=True) > 0).float()], dim=1) # (batchsize, R+1)
                #print(label_array)
                #sys.stdout.flush()
                label_array = label_array.argmax(1) # (batchsize,)
                loss = loss_func(scores, label_array.to(self.device))



            """End"""
            loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
            
            # linear warmup 
            if self.config["warmup"] > 0 and i < self.config["warmup"]:
                #print(float(i) / self.config["warmup"])
                for p in self.opt.param_groups:
                    p['lr'] = self.config["learning_rate"] * float(i) / self.config["warmup"]

            self.opt.step()

            rolling_loss.append(float(loss.detach().cpu()))
            #self.logger.debug(f"training {loss.detach().cpu()}")
            # print training loss 
            if i % 100 < self.config["train_batch_size"]:
                self.logger.info(f"{i}-th {i%100} loss: {np.mean(rolling_loss)}")
                if self.config["wandb"]:
                    wandb.log({'iteration': i, 'loss': np.mean(rolling_loss)})
                if i < self.config["warmup"]:
                    self.logger.info(f"{i}-th {i%100} lr = {self.config['learning_rate'] * float(i) / self.config['warmup']}")
                rolling_loss = []

            # evaluate on dev set
            if i % self.config["log_interval"] <= self.config["train_batch_size"]:
                macro_perf, micro_perf, per_rel_perf = self.test(test=False) # per_rel_perf is a list of length of (number of relation types + 1)
                
                if micro_perf["F"] > best_metric or i == self.config["train_batch_size"]:
                    if self.config["wandb"]:
                        wandb.log({'Micro F1 dev': micro_perf["F"], "Micro P dev": micro_perf["P"], "Micro R dev": micro_perf["R"]})
                        wandb.log({'Macro F1 dev': macro_perf["F"], "Macro P dev": macro_perf["P"], "Macro R dev": macro_perf["R"]})
                    best_metric = micro_perf["F"]
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1 (a new best) ={macro_perf['F']}")
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1 (a new best) ={micro_perf['F']}")
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        self.logger.info(f"{i * 100 / len(self.data)} % DEV (a new best): {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt} (threshold not used for multiclass)")
                        best_metric_threshold[rel_name] = tt
                    patience = 0
                    self.save_model(best_metric_threshold)
                    
                else:
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV: Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
                    self.logger.info(f"{i * 100 / len(self.data)} % DEV: Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        self.logger.info(f"{i * 100 / len(self.data)} % DEV: {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt} (threshold not used for multiclass)")
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
        if self.config["wandb"]:
            wandb.log({'Micro F1 test': micro_perf["F"], "Micro P test": micro_perf["P"], "Micro R test": micro_perf["R"]})
            wandb.log({'Macro F1 test': macro_perf["F"], "Macro P test": macro_perf["P"], "Macro R test": macro_perf["R"]})
        self.logger.info(f"Test: Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
        self.logger.info(f"Test: Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
        for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
            self.logger.info(f"TEST: {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt}")
            
        
        if self.config["wandb"]:
            wandb.finish()

    

    def test(self, test=False, best_metric_threshold=None):

        if test==True:
            self.logger.debug("This is testing")
            # if in test mode, use existing metric thresholds
            assert best_metric_threshold!=None, "evaluation on test data requires best_metric_threshold"
            # output results to a seperate file
            fout = open(os.path.join(self.config["output_path"], "test.results.json"), "w")
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
            if self.config["multi_label"] == True:
                scores = np.zeros((len(data), len(self.data.relation_map))) # (num_test, R)
            else: 
                scores = np.zeros((len(data), len(self.data.relation_map)+1)) # (num_test, R + 1)

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
                    e1_indicator = torch.tensor(np.array(e1_indicators), dtype=torch.float).to(self.device)
                    e2_indicator = torch.tensor(np.array(e2_indicators), dtype=torch.float).to(self.device)

                    ep_mask = np.full((len(input_array), self.data.max_text_length, self.data.max_text_length), -1e8)
                    for j in range(len(input_array)):
                        ep_outer = 1 - np.outer(e1_indicators[j], e2_indicators[j])
                        ep_mask[j] = ep_mask[j] * ep_outer
                    ep_mask = torch.tensor(ep_mask, dtype=torch.float).to(self.device)
                    
                    
                    score = self.model(input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator).detach().cpu().numpy() # (b, R) or (b, R+1)
                    #print(input_ids.shape, ep_mask.shape, score.shape, i+1-len(input_array), i+1, len(labels))
                    sys.stdout.flush()

                    scores[(i+1-len(input_array)):(i+1), :] = score
                    labels[(i+1-len(input_array)):(i+1), :] = np.array(label_array)

                    if test == True:
                        
                        # in test mode, save predictions for each data point (docid, e1, e2)
                        if self.config["multi_label"] == True:
                            prediction = (score > threshold_vec) # (b, R)
                        else: 
                            #score = np.concatenate([score, -logsumexp(score, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                            #score = np.concatenate([score, -np.max(score, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                            # if multi_class, choose argmax when the model predicts multiple labels 
                            prediction = np.zeros_like(score) # (num_test, R + 1)
                            # predictions[np.arange(scores.shape[0]), np.argmax((scores > threshold_vec) * (scores + 1e10), 1)] = 1
                            prediction[np.arange(score.shape[0]), np.argmax(score, 1)] = 1
                            prediction = prediction[:, :-1] # (batchsize, R), become NA if all-zero

                        for j in range(len(input_array)):
                            predict_names = []
                            for k in list(np.where(prediction[j] == 1)[0]):
                                predict_names.append(self.data.relation_name[k])
                            label_names = []
                            for k in list(np.where(label_array[j] == 1)[0]):
                                label_names.append(self.data.relation_name[k])
                            score_dict = {}
                            for k, scr in enumerate(list(score[j])):
                                if k not in self.data.relation_name:
                                    score_dict["NA"] = float(scr)
                                else:
                                    score_dict[self.data.relation_name[k]] = float(scr)
                            fout_json["predictions"].append({"docid": docid[j], "e1": e1id[j], "e2": e2id[j], "label_names": label_names, "predictions": predict_names, "scores": score_dict})
                    
                    input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id = [], [], [], [], [], [], [], []

        def calculate_metrics(predictions, labels, eval_NA=False):
            # predictions: (N, R), does not include NA in R
            # labels: (N, R), one and zeros, does not include NA in R
            TPs = predictions * labels #(N, R)
            TP = TPs.sum()
            P = predictions.sum()
            T = labels.sum()

            if eval_NA == True: # True only if set to True and multi_class is True
                NA_Ps = (predictions.sum(1) == 0)
                NA_Ts = (labels.sum(1) == 0)
                NA_TPs = NA_Ps * NA_Ts
                NA_P = NA_Ps.sum()
                NA_T = NA_Ts.sum()
                NA_TP = NA_TPs.sum()
                P = P + NA_P
                T = T + NA_T
                TP = TP + NA_TP
                NA_prec = NA_TP / NA_P if NA_P != 0 else 0
                NA_recall = NA_TP / NA_T if NA_T != 0 else 0
                NA_f = 2 * NA_prec + NA_recall / (NA_prec + NA_recall) if NA_prec + NA_recall != 0 else 0

            #FP = predictions * (labels == 0.0)
            #FN = (predictions == 0.0) * labels
            
            if P == 0:
                micro_p = 0.0
            else:
                micro_p = TP / P
            if T == 0:
                micro_r = 0.0
            else:
                micro_r = TP / T
            if micro_p == 0.0 or micro_r == 0.0:
                micro_f = 0.0
            else:
                micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)

            per_rel_p = np.zeros(predictions.shape[1])
            per_rel_r = np.zeros(predictions.shape[1])
            per_rel_f = np.zeros(predictions.shape[1])
            # per relation metrics:
            for i in range(predictions.shape[1]):
                TP_ = TPs[:, i].sum()
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
            
            if eval_NA == False:
                macro_p = per_rel_p.mean()
                macro_r = per_rel_r.mean()
                macro_f = per_rel_f.mean()
            if eval_NA == True:
                num_types = predictions.shape[1] + 1
                macro_p = (per_rel_p.sum() + NA_prec) / num_types
                macro_r = (per_rel_r.sum() + NA_recall) / num_types
                macro_f = (per_rel_f.sum() + NA_f) / num_types
                

            return micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f

        if test==True:
            # in test mode, use existing thresholds, save results.
            
            if self.config["multi_label"] == True:
                predictions = (scores > threshold_vec) # (num_test, R)
            else:
                #scores = np.concatenate([scores, -logsumexp(scores, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                #scores = np.concatenate([scores, -np.max(scores, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                # if multi_class, choose argmax when the model predicts multiple labels 
                predictions = np.zeros_like(scores) # (num_test, R + 1)
                # predictions[np.arange(scores.shape[0]), np.argmax((scores > threshold_vec) * (scores + 1e10), 1)] = 1
                predictions[np.arange(scores.shape[0]), np.argmax(scores, 1)] = 1
                predictions = predictions[:, :-1] # (batchsize, R)
                

            micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f = calculate_metrics(predictions, labels, eval_NA=self.config["eval_na"])
            
            fout_json["results"]["micro"] = {"P": float(micro_p), "R": float(micro_r), "F": float(micro_f)}
            fout_json["results"]["macro"] = {"P": float(macro_p), "R": float(macro_r), "F": float(macro_f)}
            for i, rel_name in self.data.relation_name.items():
                fout_json["results"]["per_rel"][rel_name] = {"P": float(per_rel_p[i]), "R": float(per_rel_r[i]), "F": float(per_rel_f[i])}
            #print(fout_json)
            fout.write(json.dumps(fout_json, indent="\t"))
            fout.close()

        else:
            if self.config["multi_label"] == True:
                # in validation model, tune the thresholds
                for i, rel_name in self.data.relation_name.items():
                    prec_array, recall_array, threshold_array = metrics.precision_recall_curve(labels[:, i], scores[:, i])
                    f1_array = 2 * prec_array * recall_array / (prec_array + recall_array)
                    best_threshold = threshold_array[np.argmax(f1_array)]
                    threshold_vec[i] = best_threshold
                predictions = (scores > threshold_vec) # (num_test, R)
            else:
                # if multi_class, choose argmax
                #scores = np.concatenate([scores, -logsumexp(scores, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                #scores = np.concatenate([scores, -np.max(scores, axis=1)[:, None]], axis=1) # (batchsize, R+1)
                predictions = np.zeros_like(scores) # (num_test, R+1)
                predictions[np.arange(scores.shape[0]), np.argmax(scores, 1)] = 1
                predictions = predictions[:, :-1] # (batchsize, R)
                #predictions[np.arange(scores.shape[0]), np.argmax((scores > threshold_vec) * (scores + 1e10), 1)] = 1
                

            micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, per_rel_p, per_rel_r, per_rel_f = calculate_metrics(predictions, labels, eval_NA=self.config["eval_na"])


        metric = {}
        macro_perf = {"P": macro_p, "R": macro_r, "F": macro_f}
        micro_perf = {"P": micro_p, "R": micro_r, "F": micro_f}
        per_rel_perf = {}
        for i, rel_name in self.data.relation_name.items():
            per_rel_perf[rel_name] = [per_rel_p[i], per_rel_r[i], per_rel_f[i], threshold_vec[i]]
        return macro_perf, micro_perf, per_rel_perf
