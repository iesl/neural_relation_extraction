import time
import os
import logging
import json

from transformers import AutoTokenizer, AutoModel

from module.data_loader import Dataloader
from module.utils import cuda_if_available
from module.model import BiaffineNetwork, ConcatNonLinear, MinMaxBox
from module.model import HierarchyBiaffineNetwork, HierarchyConcatNonLinear, HierarchyMinMaxBox

__all__ = [
    "setup",
]

def setup(config):
    """
    Setup and return the datasets, dataloaders, model, logger, and training loop required for training.
    :param config: config dictionary, config will also be modified
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """

    if config["output_path"] == "":
        base_dir_name = "_".join([config["encoder_type"].replace("/","-"),
                                    config["score_func"],
                                    "multilabel_" + str(config["multi_label"]),
                                    "nah_" + str(config["na_hierarchy"]),
                                    "fullanno_" + str(config["full_annotation"]),
                                    "bz_" + str(config["train_batch_size"]),
                                    "length_" + str(config["max_text_length"]),
                                    "lr_" + str(config["learning_rate"])[:7],
                                    "decay_" + str(config["weight_decay"])[:7],
                                    "dim_" + str(config["dim"]),
                                    "volt_" + str(config["volume_temp"])[:7],
                                    "intt_" + str(config["intersection_temp"])[:7],
                                    "epochs_" + str(config["epochs"]),
                                    "patience_" + str(config["patience"]),
                                    "interval_" + str(config["log_interval"]),
                                    "warmup_" + str(config["warmup"])[:7],
                                    "seed_" + str(config["seed"])
                                    ])
        config["data_path"] = config["data_path"].rstrip("/")
        config["output_path"] = os.path.join(os.environ['BIORE_OUTPUT_ROOT'], "saved_models", 
                                    os.path.basename(config["data_path"]), 
                                    base_dir_name)

    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    logger = logging.getLogger("BioRE")
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper() # DEBUG, INFO
    logger.setLevel(LOGLEVEL)
    # console_logging_stream = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # console_logging_stream.setFormatter(formatter)
    # logger.addHandler(console_logging_stream)
    logging_output = logging.FileHandler(config["output_path"] + "/log")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    logging_output.setFormatter(formatter)
    logger.addHandler(logging_output)
    
    logger.info("Program started")
    logger.info(config)


    device = cuda_if_available(use_cuda=config["cuda"])
    lowercase = True if "uncased" in config["encoder_type"] else False

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_type"], use_fast=True)
    
    # add entity marker token into vocab
    if os.path.exists(config["data_path"] + "/entity_type_markers.json"):
        entity_marker_tokens = json.loads(open(config["data_path"] + "/entity_type_markers.json").read())
        if lowercase == True:
            entity_marker_tokens = [t.lower() for t in entity_marker_tokens]
        tokenizer.add_tokens(entity_marker_tokens)
    
    # setup model
    if config["score_func"] == "biaffine" and config["na_hierarchy"] == False:
        model = BiaffineNetwork(config)
    elif config["score_func"] == "biaffine" and config["na_hierarchy"] == True:
        model = HierarchyBiaffineNetwork(config)
    elif config["score_func"] == "nn" and config["na_hierarchy"] == False:
        model = ConcatNonLinear(config)
    elif config["score_func"] == "nn" and config["na_hierarchy"] == True:
        model = HierarchyConcatNonLinear(config)
    elif config["score_func"] == "box" and config["na_hierarchy"] == False:
        model = MinMaxBox(config)
    elif config["score_func"] == "box" and config["na_hierarchy"] == True:
        model = HierarchyMinMaxBox(config)

    model.to(device)
    model.encoder.resize_token_embeddings(len(tokenizer)) # create token embedding for new added markers
    

    # setup data
    time1 = time.time()
    

    data = Dataloader(config["data_path"], tokenizer, batchsize=config["train_batch_size"], shuffle=True, 
                      seed=config["seed"], max_text_length=config["max_text_length"], training=config["train"],
                      logger=logger, lowercase=lowercase, full_annotation=config["full_annotation"])
    logger.info(f"number of data points during training : {len(data)}")
    time2 = time.time()
    logger.info("Time spent loading data: %f" % (time2 - time1))

    if isinstance(config["log_interval"], float):
        config["log_interval"] = len(data) * config["log_interval"]
        logger.info(f"Log interval: {config['log_interval']}")
    if isinstance(config["warmup"], float):
        config["warmup"] = len(data) * config["warmup"]
        logger.info(f"warmup {config['warmup']}")
    if config["multi_label"] == True:
        config["eval_na"] = False


    return data, model, device, logger
