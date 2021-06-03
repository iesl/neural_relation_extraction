import numpy as np
import torch
from torch.cuda.amp import GradScaler
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule
import os
import sys
import json
import pickle
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
        self.scaler = GradScaler()
        self.opt = AdamW(
            self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], eps=1e-6
        )
        if config["warmup"] >= 0.0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=config["warmup"], num_training_steps=config["max_num_steps"])
        else:
            self.scheduler = get_constant_schedule(self.opt)

        self.bcelogitloss = torch.nn.BCEWithLogitsLoss()  # y is 1 or 0, x is 1-d logit
        # y is a non-negative integer, x is a multi dimensional logits
        self.celogitloss = torch.nn.CrossEntropyLoss()

        # for name, param in self.model.named_parameters():
        #    #if param.requires_grad:
        #    print(name)

    def save_model(self, best_metric_threshold):
        model_state_dict = self.model.state_dict()  # TODO may have device issue
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

    def save_dev_embedding(self, valid_vectors, valid_labels, valid_predictions):
        # valid_vectors: valid_size, dim
        # valid_labels: valid_size, len(relation_map)
        output_path = os.path.join(
            self.config["output_path"], 'vectors_valid.pkl')
        relation_name_list = [r for i, r in sorted(
            list(self.data.relation_name.items()), key=lambda x:x[0])]
        with open(output_path, 'wb') as f:
            if self.config["score_func"] == "cosine":
                rel_mat = self.model.relation_mat.data.detach().cpu().numpy().T  # R+1, D
                rel_mat = rel_mat / \
                    np.linalg.norm(rel_mat, ord=2, axis=1, keepdims=True)
            else:
                rel_mat = self.model.layer2.weight.detach().cpu().numpy()  # R+1, D
            output_data = {"vectors": valid_vectors, "preds": valid_predictions, "label_name": relation_name_list,
                           "labels": valid_labels, "relation_mat": rel_mat}
            pickle.dump(output_data, f)

    def wandb_logging(self, micro_perf, macro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, label="dev"):
        wandb.log(
            {'Micro F1 ' + label: micro_perf["F"], "Micro P " + label: micro_perf["P"], "Micro R " + label: micro_perf["R"]})
        wandb.log(
            {'Macro F1 ' + label: macro_perf["F"], "Macro P " + label: macro_perf["P"], "Macro R " + label: macro_perf["R"]})
        wandb.log(
            {'Categorical Acc ' + label: categ_acc, 'Categorical Macro F1 ' + label: categ_macro_perf["F"], "Categorical Macro P " + label: categ_macro_perf["P"], "Categorical Macro R " + label: categ_macro_perf["R"]})
        wandb.log(
            {'na Acc ' + label: na_acc, "not_na P " + label: not_na_perf["P"], "not_na R " + label: not_na_perf["R"], 'not_na F1 ' + label: not_na_perf["F"]})
        wandb.log(
            {"na P " + label: na_perf["P"], "na R " + label: na_perf["R"], 'na F1 ' + label: na_perf["F"]})

    def local_logging(self, micro_perf, macro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf, i, label="DEV"):
        if len(self.data) == 0:
            self.data = [""]
        self.logger.info(
            f"{i * 100 / len(self.data)} % {label}: Micro P={micro_perf['P']}, Micro R={micro_perf['R']}, Micro F1 ={micro_perf['F']}")
        self.logger.info(
            f"{i * 100 / len(self.data)} % {label}: Macro P={macro_perf['P']}, Macro R={macro_perf['R']}, Macro F1 ={macro_perf['F']}")
        self.logger.info(
            f"{i * 100 / len(self.data)} % {label}: Categorical Accuracy={categ_acc}, Categorical Macro P={categ_macro_perf['P']}, Categorical Macro R={categ_macro_perf['R']}, Categorical Macro F1 ={categ_macro_perf['F']}")
        self.logger.info(
            f"{i * 100 / len(self.data)} % {label}: not_na Accuracy={na_acc}, not_na P={not_na_perf['P']}, not_na R={not_na_perf['R']}, not_na F1 ={not_na_perf['F']}")
        self.logger.info(
            f"{i * 100 / len(self.data)} % {label} na P={na_perf['P']}, na R={na_perf['R']}, na F1 ={na_perf['F']}")
        for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
            self.logger.info(
                f"{i * 100 / len(self.data)} % {label}: {rel_name}, P={pp}, R={rr}, F1={ff}, threshold={tt} (threshold not used for multiclass)")

    def train(self):
        if self.config["wandb"]:
            wandb.init(project="re", settings=wandb.Settings(
                start_method="fork"))
            wandb.config.update(self.config, allow_val_change=True)

        self.logger.debug("This is training")

        if self.config["multi_label"] == True and self.config["score_func"] != "box":
            def loss_func(input, target): return self.bcelogitloss(
                input, target)
        elif self.config["multi_label"] == True and self.config["score_func"] == "box":
            def loss_func(input, target): return -(target * input +
                                                   (1 - target) * log1mexp(input)).mean()  # bce log loss
        elif self.config["multi_label"] == False:

            if self.config["classifier"] == "softmax":
                # regular softmax cross entropy loss
                def loss_func(input, target):
                    # input: batchsize, R+1
                    # target: batchsize, R
                    target = torch.cat(
                        [target, 1 - (target.sum(1, keepdim=True) > 0).float()], dim=1)  # (batchsize, R + 1)
                    input = self.config["rescale_factor"] * input
                    target = target.argmax(1)
                    return self.celogitloss(input, target)  # input are logits

            elif self.config["classifier"] == "softmax_margin":
                # softmax with positive margin
                # https://arxiv.org/abs/1801.09414  equation (4)
                def loss_func(input, target):
                    # input: batchsize, R+1
                    # target: batchsize, R
                    target = torch.cat(
                        [target, 1 - (target.sum(1, keepdim=True) > 0).float()], dim=1)  # (batchsize, R + 1)
                    input = self.config["rescale_factor"] * \
                        (input - target * self.config["margin"])
                    target = target.argmax(1)
                    return self.celogitloss(input, target)

            elif self.config["classifier"] == "pairwise_margin":
                # ranking loss from "Classifying Relations by Ranking with Convolutional Neural Networks"
                def loss_func(input, target):
                    # input: batchsize, R+1
                    # target: batchsize, R
                    target = torch.cat(
                        [target, 1 - (target.sum(1, keepdim=True) > 0).float()], dim=1)  # (batchsize, R + 1)
                    pos_input = (target.to(self.device) *
                                 input).sum(1)  # (batchsize, )
                    neg_input = (1 - target.to(self.device)) * \
                        input  # (batchsize, R + 1)

                    # positive score should be greater than pairwise_margin_pos,
                    # negative score should be less than -pairwise_margin_neg
                    # for dot product, pairwise_margin_pos could be 2.5, pairwise_margin_neg could be 0.5
                    # for cosine, pairwise_margin_pos could be 0.3, pairwise_margin_neg could be 0.1
                    loss = torch.nn.functional.softplus(self.config["rescale_factor"] * (self.config["margin_pos"] - pos_input)).mean(
                    ) + torch.nn.functional.softplus(self.config["rescale_factor"] * (self.config["margin_neg"] + neg_input)).mean()  # (batchsize, )
                    return loss

            elif self.config["classifier"] == "pairwise_margin_na":
                # ranking loss from "Classifying Relations by Ranking with Convolutional Neural Networks"
                def loss_func(input, target):
                    # input: batchsize, R+1
                    # target: batchsize, R
                    label_not_NA = (target.sum(1) != 0).to(
                        self.device)  # (batchsize, )
                    pos_input = (target.to(self.device) *
                                 input[:, :-1]).sum(1)  # (batchsize, )
                    neg_input = (1 - target.to(self.device)) * \
                        input[:, :-1]  # (batchsize, R)

                    # positive score should be greater than pairwise_margin_pos,
                    # negative score should be less than -pairwise_margin_neg
                    # for dot product, pairwise_margin_pos could be 2.5, pairwise_margin_neg could be 0.5
                    # for cosine, pairwise_margin_pos could be 0.3, pairwise_margin_neg could be 0.1
                    loss = (label_not_NA * torch.nn.functional.softplus(self.config["rescale_factor"] * (self.config["margin_pos"] - pos_input))).mean(
                    ) + torch.nn.functional.softplus(self.config["rescale_factor"] * (self.config["margin_neg"] + neg_input)).mean()  # (batchsize, )
                    return loss

        best_metric = -1
        best_metric_threshold = {}
        patience = 0
        rolling_loss = []
        max_step = self.config["epochs"] * len(self.data)
        self.model.zero_grad()
        for i, (input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator, label_array) in iter(self.data):
            self.model.train(True)

            """Loss"""
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            ep_mask = ep_mask.to(self.device)
            e1_indicator = e1_indicator.to(self.device)
            e2_indicator = e2_indicator.to(self.device)
            scores, _ = self.model(input_ids, token_type_ids, attention_mask, ep_mask,
                                   e1_indicator, e2_indicator)  # (batchsize, R) or (batchsize, R+1)
            if self.config["multi_label"] == True:
                loss = loss_func(scores, label_array.to(self.device))
            else:
                # (batchsize,)
                loss = loss_func(scores, label_array.to(self.device))

            """back prop"""
            loss = loss / self.config["grad_accumulation_steps"]
            self.scaler.scale(loss).backward()

            for param in self.model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()

            step = i / (self.config["train_batch_size"])
            if step % self.config["grad_accumulation_steps"] == 0:
                if self.config['max_grad_norm'] > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['max_grad_norm'])  # clip gradient norm
                self.scaler.step(self.opt)
                sys.stdout.flush()
                self.scaler.update()
                self.scheduler.step()
                self.model.zero_grad()

            """End"""
            rolling_loss.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                self.logger.info(
                    f"{i}-th example loss: {np.mean(rolling_loss)}")
                # print(f"{i}-th example loss: {np.mean(rolling_loss)}")
                if self.config["wandb"]:
                    wandb.log({'step': step, 'loss': np.mean(rolling_loss)})
                rolling_loss = []

            # evaluate on dev set (if out-performed, evaluate on test as well)
            if 0 < i % self.config["log_interval"] <= self.config["train_batch_size"]:
                # print(i, self.config["log_interval"], i %
                #      self.config["log_interval"], self.config["train_batch_size"])
                self.model.eval()

                # replace relation matrix with the center of instance of same label.
                if self.config["score_func"] == "cosine":
                    label_centers = self.calculate_centers(
                        self.data.label2train)  # R, D; normalized
                    self.model.relation_mat.data[:, :-1] = torch.tensor(
                        label_centers).float().T.to(self.device)

                # per_rel_perf is a list of length of (number of relation types + 1)
                macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf, valid_vectors, valid_labels, valid_preds, _ = self.test(
                    test_mode=False)
                self.logger.info(f'val: {micro_perf["F"]}, {not_na_perf["F"]}')

                if micro_perf["F"] > best_metric or i == self.config["train_batch_size"]:
                    if self.config["wandb"]:
                        self.wandb_logging(micro_perf, macro_perf, categ_acc,
                                           categ_macro_perf, na_acc, not_na_perf, na_perf, label="dev")

                    best_metric = micro_perf["F"]
                    self.local_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf, na_acc,
                                       not_na_perf, na_perf, per_rel_perf, i, label="DEV (a new best)")

                    #     best_metric_threshold[rel_name] = tt
                    for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
                        best_metric_threshold[rel_name] = tt
                    patience = 0
                    self.save_model(best_metric_threshold)

                    if self.config["save_embedding"] == True and self.config["score_func"] in ["dot", "cosine"]:
                        self.save_dev_embedding(
                            valid_vectors, valid_labels, valid_preds)

                    # evaluate on test set
                    macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf, _, _, _, _ = self.test(
                        test_mode=True, best_metric_threshold=best_metric_threshold)
                    self.logger.info(f'test: {micro_perf["F"]}, {not_na_perf["F"]}')
                    sys.stdout.flush()
                    if self.config["wandb"]:
                        self.wandb_logging(micro_perf, macro_perf, categ_acc,
                                           categ_macro_perf, na_acc, not_na_perf, na_perf, label="test")
                    self.local_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                                       na_acc, not_na_perf, na_perf, per_rel_perf, i, label="TEST")

                else:
                    self.local_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                                       na_acc, not_na_perf, na_perf, per_rel_perf, i, label="DEV")

                    patience += 1

            # early stop
            if patience > self.config["patience"]:
                self.logger.info("triggers early stop; ended")
                break
            if i > max_step:
                self.logger.info("exceeds maximum steps; ended")
                break

        # after finished, load best model and evaluate on test again
        # best_metric_threshold = self.load_model()
        # self.model.eval()
        # macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf, _, _, _ = self.test(
        #     test=True, best_metric_threshold=best_metric_threshold)
        # print("test", micro_perf["F"])
        # sys.stdout.flush()
        # if self.config["wandb"]:
        #     self.wandb_logging(micro_perf, macro_perf, categ_acc,
        #                        categ_macro_perf, na_acc, not_na_perf, na_perf, label="test")

        # self.local_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
        #                    na_acc, not_na_perf, na_perf, per_rel_perf, 0, label="TEST")

        if self.config["wandb"]:
            wandb.finish()

    def calculate_metrics(self, predictions, predictions_categ, labels):
        # calcuate metrics given prediction and labels
        # predictions: (N, R), does not include NA in R
        # labels: (N, R), one and zeros, does not include NA in R
        # predictions_categ: (N, R), contains predictions for calculating performance of categorical classifier (exclude NA)
        TPs = predictions * labels  # (N, R)
        TP = TPs.sum()
        P = predictions.sum()
        T = labels.sum()

        micro_p = TP / P if P != 0 else 0
        micro_r = TP / T if T != 0 else 0
        micro_f = 2 * micro_p * micro_r / \
            (micro_p + micro_r) if micro_p + micro_r > 0 else 0

        categ_TPs = predictions_categ * labels
        categ_TP = categ_TPs.sum()
        # exludes instance whose label is NA
        categ_Ps = (predictions_categ * (labels.sum(1) > 0)[:, None])

        categ_acc = categ_TP / T if T != 0 else 0

        not_NA_Ps = (predictions.sum(1) > 0)
        not_NA_Ts = (labels.sum(1) > 0)
        not_NA_TPs = not_NA_Ps * not_NA_Ts
        not_NA_P = not_NA_Ps.sum()
        not_NA_T = not_NA_Ts.sum()
        not_NA_TP = not_NA_TPs.sum()
        not_NA_prec = not_NA_TP / not_NA_P if not_NA_P != 0 else 0
        not_NA_recall = not_NA_TP / not_NA_T if not_NA_T != 0 else 0
        not_NA_f = 2 * not_NA_prec * not_NA_recall / \
            (not_NA_prec + not_NA_recall) if not_NA_prec + \
            not_NA_recall > 0 else 0

        not_NA_acc = (not_NA_Ps == not_NA_Ts).mean()

        NA_Ps = (predictions.sum(1) == 0)
        NA_Ts = (labels.sum(1) == 0)
        NA_TPs = NA_Ps * NA_Ts
        NA_P = NA_Ps.sum()
        NA_T = NA_Ts.sum()
        NA_TP = NA_TPs.sum()
        NA_prec = NA_TP / NA_P if NA_P != 0 else 0
        NA_recall = NA_TP / NA_T if NA_T != 0 else 0
        NA_f = 2 * NA_prec * NA_recall / \
            (NA_prec + NA_recall) if NA_prec + NA_recall > 0 else 0

        per_rel_p = np.zeros(predictions.shape[1])
        per_rel_r = np.zeros(predictions.shape[1])
        per_rel_f = np.zeros(predictions.shape[1])
        categ_per_rel_p = np.zeros(predictions.shape[1])
        categ_per_rel_r = np.zeros(predictions.shape[1])
        categ_per_rel_f = np.zeros(predictions.shape[1])
        # per relation metrics:
        for i in range(predictions.shape[1]):
            TP_ = TPs[:, i].sum()
            P_ = predictions[:, i].sum()
            T_ = labels[:, i].sum()
            categ_TP_ = categ_TPs[:, i].sum()
            categ_P_ = categ_Ps[:, i].sum()

            # if no such relation in the test data, recall = 0
            per_rel_r[i] = TP_ / T_ if T_ != 0 else 0
            categ_per_rel_r[i] = categ_TP_ / T_ if T_ != 0 else 0

            # if no such relation in the prediction, precision = 0
            per_rel_p[i] = TP_ / P_ if P_ != 0 else 0

            # if no such relation in the prediction, precision = 0
            categ_per_rel_p[i] = categ_TP_ / categ_P_ if categ_P_ != 0 else 0

            per_rel_f[i] = 2 * per_rel_p[i] * per_rel_r[i] / \
                (per_rel_p[i] + per_rel_r[i]) if per_rel_p[i] + \
                per_rel_r[i] > 0 else 0

            categ_per_rel_f[i] = 2 * categ_per_rel_p[i] * categ_per_rel_r[i] / \
                (categ_per_rel_p[i] + categ_per_rel_r[i]
                 ) if categ_per_rel_p[i] + categ_per_rel_r[i] > 0 else 0

        macro_p = per_rel_p.mean()
        macro_r = per_rel_r.mean()
        macro_f = per_rel_f.mean()

        categ_macro_p = categ_per_rel_p.mean()
        categ_macro_r = categ_per_rel_r.mean()
        categ_macro_f = categ_per_rel_f.mean()

        results = {
            "micro_p": micro_p,
            "micro_r": micro_r,
            "micro_f": micro_f,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f": macro_f,
            "categ_acc": categ_acc,
            "categ_macro_p": categ_macro_p,
            "categ_macro_r": categ_macro_r,
            "categ_macro_f": categ_macro_f,
            "na_acc": not_NA_acc,
            "not_na_p": not_NA_prec,
            "not_na_r": not_NA_recall,
            "not_na_f": not_NA_f,
            "na_p": NA_prec,
            "na_r": NA_recall,
            "na_f": NA_f,
            "per_rel_p": per_rel_p,
            "per_rel_r": per_rel_r,
            "per_rel_f": per_rel_f,
            "categ_per_rel_p": categ_per_rel_p,
            "categ_per_rel_r": categ_per_rel_r,
            "categ_per_rel_f": categ_per_rel_f,
        }

        return results

    def test(self, test_mode=False, best_metric_threshold=None):

        # load data
        if test_mode == True:
            self.logger.debug("This is testing")
            # if in test mode, use existing metric thresholds
            assert best_metric_threshold != None, "evaluation on test data requires best_metric_threshold"
            # output results to a seperate file
            fout = open(os.path.join(
                self.config["output_path"], "test.results.json"), "w")
            fout_json = {"threshold": {}, "predictions": [],
                         "results": {"macro": {}, "micro": {}, "per_rel": {}}}

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
        sys.stdout.flush()

        # Infer scores for valid/test data
        with torch.no_grad():
            if self.config["multi_label"] == True:
                # (num_test, R)
                scores = np.zeros((len(data), len(self.data.relation_map)))
            else:
                # (num_test, R + 1)
                scores = np.zeros((len(data), len(self.data.relation_map)+1))

            # (num_test, R)
            labels = np.zeros((len(data), len(self.data.relation_map)))
            vectors = np.zeros((len(data), self.config["dim"]))
            docids = []
            self.logger.debug(f"length of data: {len(data)}")
            input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id, input_lengths = [
            ], [], [], [], [], [], [], [], []

            for i, dict_ in tqdm(enumerate(data)):
                input_array.append(dict_["input"])
                pad_array.append(dict_["pad"])
                label_array.append(dict_["label_vector"])
                e1_indicators.append(dict_["e1_indicators"])
                e2_indicators.append(dict_["e2_indicators"])
                docid.append(dict_["docid"])
                docids.append(dict_["docid"])
                e1id.append(dict_["e1"])
                e2id.append(dict_["e2"])
                input_lengths.append(dict_["input_length"])

                if len(input_array) == self.config["test_batch_size"] or i == len(data) - 1:
                    max_length = np.max(input_lengths)
                    input_ids = torch.tensor(
                        np.array(input_array)[:, :max_length], dtype=torch.long).to(self.device)
                    token_type_ids = torch.zeros_like(
                        input_ids[:, :max_length], dtype=torch.long).to(self.device)
                    attention_mask = torch.tensor(
                        np.array(pad_array)[:, :max_length], dtype=torch.long).to(self.device)
                    e1_indicator = torch.tensor(
                        np.array(e1_indicators)[:, :max_length], dtype=torch.float).to(self.device)
                    e2_indicator = torch.tensor(
                        np.array(e2_indicators)[:, :max_length], dtype=torch.float).to(self.device)

                    ep_mask = np.full(
                        (len(input_array), self.data.max_text_length, self.data.max_text_length), -1e20)
                    for j in range(len(input_array)):
                        ep_outer = 1 - \
                            np.outer(e1_indicators[j], e2_indicators[j])
                        ep_mask[j] = ep_mask[j] * ep_outer
                    ep_mask = torch.tensor(
                        ep_mask[:, :max_length, :max_length], dtype=torch.float).to(self.device)

                    score, vector = self.model(input_ids, token_type_ids, attention_mask, ep_mask,
                                               e1_indicator, e2_indicator)  # (b, R) or (b, R+1)
                    score = score.detach().cpu().numpy()
                    if vector != None:
                        vector = vector.detach().cpu().numpy()
                        vectors[(i+1-len(input_array)):(i+1), :] = vector

                    # print(input_ids.shape, ep_mask.shape, score.shape, i+1-len(input_array), i+1, len(labels))
                    sys.stdout.flush()

                    scores[(i+1-len(input_array)):(i+1), :] = score
                    labels[(i+1-len(input_array)):(i+1),
                           :] = np.array(label_array)

                    if test_mode == True:

                        # in test mode, save predictions for each data point (docid, e1, e2)
                        if self.config["multi_label"] == True:
                            prediction = (score > threshold_vec)  # (b, R)
                        else:

                            if self.config["classifier"] == "pairwise_margin_na":
                                # prediction from "Classifying Relations by Ranking with Convolutional Neural Networks""
                                prediction_categ = np.zeros_like(
                                    score[:, :-1])  # (b, R)
                                prediction_categ[np.arange(
                                    score.shape[0]), np.argmax(score[:, :-1], 1)] = 1

                                # if none of the scores above zero, prediction will be NA
                                # (batchsize, R), predicts NA if prediction[:, :-1] is all-zero
                                prediction = np.where((score[:, :-1] > 0).sum(
                                    1)[:, None] > 0, prediction_categ, np.zeros_like(prediction_categ))

                            else:
                                prediction = np.zeros_like(
                                    score)  # (num_test, R + 1)
                                prediction[np.arange(
                                    score.shape[0]), np.argmax(score, 1)] = 1

                                # (batchsize, R), predicts NA if prediction[:, :-1] is all-zero
                                prediction = prediction[:, :-1]

                        for j in range(len(input_array)):
                            predict_names = []
                            for k in list(np.where(prediction[j] == 1)[0]):
                                predict_names.append(
                                    self.data.relation_name[k])
                            label_names = []
                            for k in list(np.where(label_array[j] == 1)[0]):
                                label_names.append(self.data.relation_name[k])
                            score_dict = {}
                            for k, scr in enumerate(list(score[j])):
                                if k not in self.data.relation_name:
                                    score_dict["NA"] = float(scr)
                                else:
                                    score_dict[self.data.relation_name[k]] = float(
                                        scr)
                            fout_json["predictions"].append(
                                {"docid": docid[j], "e1": e1id[j], "e2": e2id[j], "label_names": label_names, "predictions": predict_names, "scores": score_dict})

                    input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id, input_lengths = [
                    ], [], [], [], [], [], [], [], []

        # calculate metrics for valid/test data
        if test_mode == True:
            # in test mode, use existing thresholds, save results.

            if self.config["multi_label"] == True:
                predictions = (scores > threshold_vec)  # (num_test, R)
                predictions_categ = predictions
            else:
                # if multi_class, choose argmax when the model predicts multiple labels

                if self.config["classifier"] == "pairwise_margin_na":
                    # prediction from "Classifying Relations by Ranking with Convolutional Neural Networks""
                    predictions = np.zeros_like(scores)  # (num_test, R + 1)
                    predictions_categ = np.zeros_like(
                        scores[:, :-1])  # (num_test, R)
                    predictions_categ[np.arange(
                        scores.shape[0]), np.argmax(scores[:, :-1], 1)] = 1

                    # if none of the scores above zero, prediction will be NA
                    # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                    predictions = np.where((scores[:, :-1] > 0).sum(
                        1)[:, None] > 0, predictions_categ, np.zeros_like(predictions_categ))

                else:
                    predictions = np.zeros_like(scores)  # (num_test, R + 1)
                    predictions[np.arange(scores.shape[0]),
                                np.argmax(scores, 1)] = 1

                    # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                    predictions = predictions[:, :-1]

                    predictions_categ = np.zeros_like(scores)[:, :-1]
                    predictions_categ[np.arange(
                        scores.shape[0]), np.argmax(scores[:, :-1], 1)] = 1

            results = self.calculate_metrics(
                predictions, predictions_categ, labels)

            fout_json["results"]["micro"] = {"P": float(
                results["micro_p"]), "R": float(results["micro_r"]), "F": float(results["micro_f"])}
            fout_json["results"]["macro"] = {"P": float(
                results["macro_p"]), "R": float(results["macro_r"]), "F": float(results["macro_f"])}
            fout_json["results"]["categ_acc"] = float(results["categ_acc"])
            fout_json["results"]["categ_macro"] = {"P": float(
                results["categ_macro_p"]), "R": float(results["categ_macro_r"]), "F": float(results["categ_macro_f"])}
            fout_json["results"]["na_acc"] = float(results["na_acc"])
            fout_json["results"]["not_na"] = {"P": float(
                results["not_na_p"]), "R": float(results["not_na_r"]), "F": float(results["not_na_f"])}
            fout_json["results"]["na"] = {"P": float(
                results["na_p"]), "R": float(results["na_r"]), "F": float(results["na_f"])}
            for i, rel_name in self.data.relation_name.items():
                fout_json["results"]["per_rel"][rel_name] = {"P": float(
                    results["per_rel_p"][i]), "R": float(results["per_rel_r"][i]), "F": float(results["per_rel_f"][i])}

            # add conditional probability between each pair of relation types:
            if self.config["score_func"] == "box":
                log_cond_prob, log_marginal_prob = self.model.transition_matrix()
                log_cond_prob = log_cond_prob.detach().cpu().tolist()
                log_marginal_prob = log_marginal_prob.detach().cpu().tolist()
                fout_json["transition_matrix"] = {
                    "log_conditional": log_cond_prob, "log_marginal": log_marginal_prob}

        else:
            if self.config["multi_label"] == True:
                # in validation model, tune the thresholds
                for i, rel_name in self.data.relation_name.items():
                    prec_array, recall_array, threshold_array = metrics.precision_recall_curve(
                        labels[:, i], scores[:, i])
                    # print(prec_array, recall_array, prec_array + recall_array > 0)
                    prec_array_ = np.where(
                        prec_array + recall_array > 0, prec_array, np.ones_like(prec_array))
                    f1_array = np.where(prec_array + recall_array > 0, 2 * prec_array * recall_array / (
                        prec_array_ + recall_array), np.zeros_like(prec_array))
                    best_threshold = threshold_array[np.argmax(f1_array)]
                    threshold_vec[i] = best_threshold
                predictions = (scores > threshold_vec)  # (num_test, R)
                predictions_categ = predictions
            else:
                # if multi_class, choose argmax

                if self.config["classifier"] == "pairwise_margin_na":
                    # prediction from "Classifying Relations by Ranking with Convolutional Neural Networks""
                    predictions = np.zeros_like(scores)  # (num_test, R + 1)
                    predictions_categ = np.zeros_like(
                        scores[:, :-1])  # (num_test, R)
                    predictions_categ[np.arange(
                        scores.shape[0]), np.argmax(scores[:, :-1], 1)] = 1

                    # if none of the scores above zero, prediction will be NA
                    # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                    predictions = np.where((scores[:, :-1] > 0).sum(
                        1)[:, None] > 0, predictions_categ, np.zeros_like(predictions_categ))

                else:
                    predictions = np.zeros_like(scores)  # (num_test, R + 1)
                    predictions[np.arange(scores.shape[0]),
                                np.argmax(scores, 1)] = 1

                    # (num_test, R), predicts NA if prediction[:, :-1] is all-zero
                    predictions = predictions[:, :-1]

                    predictions_categ = np.zeros_like(scores)[:, :-1]
                    predictions_categ[np.arange(
                        scores.shape[0]), np.argmax(scores[:, :-1], 1)] = 1

            results = self.calculate_metrics(
                predictions, predictions_categ, labels)

        macro_perf = {"P": results["macro_p"],
                      "R": results["macro_r"], "F": results["macro_f"]}
        micro_perf = {"P": results["micro_p"],
                      "R": results["micro_r"], "F": results["micro_f"]}
        categ_macro_perf = {"P": results["categ_macro_p"],
                            "R": results["categ_macro_r"], "F": results["categ_macro_f"]}
        not_na_perf = {
            "P": results["not_na_p"], "R": results["not_na_r"], "F": results["not_na_f"]}
        na_perf = {"P": results["na_p"],
                   "R": results["na_r"], "F": results["na_f"]}
        per_rel_perf = {}
        for i, rel_name in self.data.relation_name.items():
            per_rel_perf[rel_name] = [results["per_rel_p"][i],
                                      results["per_rel_r"][i], results["per_rel_f"][i], threshold_vec[i]]

        # for each relation, find top 100 docs that are false positive and have largest scores.
        rel2docs = {}
        for r in range(len(self.data.relation_map)):
            label_r = labels[:, r]  # (num_test)
            score_r = scores[:, r]  # (num_test)
            predict_r = predictions[:, r]  # (num_test)
            new_score_r = np.where(predict_r != label_r,
                                   score_r, - 1e8 * np.ones_like(score_r))
            sorted_ids = np.argsort(new_score_r)[::-1][:100]
            rel2docs[self.data.relation_name[r]] = [
                (docids[i], score_r[i]) for i in list(sorted_ids)]

        if test_mode == True:
            fout_json["false_positives"] = rel2docs
            fout.write(json.dumps(fout_json, indent="\t"))
            fout.close()

        return macro_perf, micro_perf, results["categ_acc"], categ_macro_perf, results["na_acc"], not_na_perf, na_perf, per_rel_perf, vectors, labels, predictions, rel2docs

    def calculate_centers(self, label2train):
        # label2train: dictionary
        # Infer scores for valid/test data
        relation_mat = np.zeros(
            (len(label2train), int(self.config["dim"])))  # R, D

        with torch.no_grad():
            for label, data in label2train.items():
                if label in self.data.relation_map:
                    labelid = self.data.relation_map[label]
                else:
                    continue
                vectors = np.zeros((len(data), self.config["dim"]))
                input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id, input_lengths = [
                ], [], [], [], [], [], [], [], []
                for i, dict_ in enumerate(data):
                    input_array.append(dict_["input"])
                    pad_array.append(dict_["pad"])
                    label_array.append(dict_["label_vector"])
                    e1_indicators.append(dict_["e1_indicators"])
                    e2_indicators.append(dict_["e2_indicators"])
                    docid.append(dict_["docid"])
                    e1id.append(dict_["e1"])
                    e2id.append(dict_["e2"])
                    input_lengths.append(dict_["input_length"])

                    if len(input_array) == self.config["test_batch_size"] or i == len(data) - 1:

                        max_length = np.max(input_lengths)
                        input_ids = torch.tensor(
                            np.array(input_array)[:, :max_length], dtype=torch.long).to(self.device)
                        token_type_ids = torch.zeros_like(
                            input_ids[:, :max_length], dtype=torch.long).to(self.device)
                        attention_mask = torch.tensor(
                            np.array(pad_array)[:, :max_length], dtype=torch.long).to(self.device)
                        e1_indicator = torch.tensor(
                            np.array(e1_indicators)[:, :max_length], dtype=torch.float).to(self.device)
                        e2_indicator = torch.tensor(
                            np.array(e2_indicators)[:, :max_length], dtype=torch.float).to(self.device)

                        ep_mask = np.full(
                            (len(input_array), self.data.max_text_length, self.data.max_text_length), -1e20)
                        ep_outer = 1 - \
                            np.outer(e1_indicators[0], e2_indicators[0])
                        ep_mask[0] = ep_mask[0] * ep_outer
                        ep_mask = torch.tensor(
                            ep_mask[:, :max_length, :max_length], dtype=torch.float).to(self.device)

                        _, vector = self.model(input_ids, token_type_ids, attention_mask, ep_mask,
                                               e1_indicator, e2_indicator)  # (b, R) or (b, R+1)
                        vector = vector.detach().cpu().numpy()
                        vectors[(i+1-len(input_array)):(i+1), :] = vector

                        input_array, pad_array, label_array, e1_indicators, e2_indicators, docid, e1id, e2id, input_lengths = [
                        ], [], [], [], [], [], [], [], []

                average_vector = np.mean(vectors, axis=0)
                average_vector = average_vector / \
                    np.linalg.norm(average_vector, ord=2)
                relation_mat[int(labelid)] = average_vector
        self.logger.info("recalculated centers")
        return relation_mat
