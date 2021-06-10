import time
import os
import logging
import json

from transformers import AutoTokenizer, AutoModel

from module.data_loader import Dataloader
from module.utils import cuda_if_available
from module.model import BiaffineNetwork, ConcatNonLinear, ConcatNonLinearNormalized, MinMaxBox

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
        base_dir_name = "_".join([
            config["data_path"].strip("/").split("/")[-1],
            "fullanno_" +
            str(config["full_annotation"]),
            config["encoder_type"].replace("/", "-"),
            config["score_func"],
            config["classifier"],
            "multilabel_" + str(config["multi_label"]),
            "lr_" + str(config["learning_rate"])[:6],
            "dim_" + str(config["dim"]),
            "rescale_" +
            str(config["rescale_factor"])[:5],
            "marg_" + str(config["margin"])[:5],
            "margpos_" + str(config["margin_pos"])[:5],
            "margneg_" + str(config["margin_neg"])[:5],
            "contrastw_" + str(config["contrastive_weight"])[:5],
            "perturbw_" + str(config["perturb_weight"])[:5],
            "bz_" + str(config["train_batch_size"]),
            "acc_" +
            str(config["grad_accumulation_steps"]),
            "len_" + str(config["max_text_length"]),
            "decay_" + str(config["weight_decay"])[:6],
            "vt_" + str(config["volume_temp"])[:6],
            "it_" +
            str(config["intersection_temp"])[:6],
        ])
        config["data_path"] = config["data_path"].rstrip("/")
        config["output_path"] = os.path.join(os.environ['BIORE_OUTPUT_ROOT'], "saved_models",
                                             os.path.basename(
                                                 config["data_path"]),
                                             base_dir_name)

    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    logger = logging.getLogger("BioRE")
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()  # DEBUG, INFO
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

    if config['deepspeed']:
        device = cuda_if_available(use_cuda=config["cuda"])
        # # - must be run very last in arg parsing, since it will use a lot of these settings.
        # # - must be run before the model is created.
        # from transformers.integrations import DeepSpeedConfigHF

        # # will be used later by the Trainer (leave self.deepspeed unmodified in case a user relies on it not to be modified)
        # class seudoclass():
        #     def __init__(self, deepspeed):
        #         self.deepspeed = deepspeed
        # seudo = seudoclass(config["deepspeed"])

        # deepspeed_config_hf = DeepSpeedConfigHF(seudo)

        # from transformers.integrations import is_deepspeed_available

        # if not is_deepspeed_available():
        #     raise ImportError(
        #         "--deepspeed requires deepspeed: `pip install deepspeed`.")
        # import deepspeed

        # deepspeed.init_distributed()

        # # workaround for setups like notebooks where the launcher can't be used,
        # # but deepspeed requires a dist env.
        # # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
        # local_rank = int(config["local_rank"])

        # device = torch.device("cuda", local_rank)
    else:
        device = cuda_if_available(use_cuda=config["cuda"])
    lowercase = True if "uncased" in config["encoder_type"] else False

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_type"], use_fast=True)

    # add entity marker token into vocab
    if os.path.exists(config["data_path"] + "/entity_type_markers.json"):
        entity_marker_tokens = json.loads(
            open(config["data_path"] + "/entity_type_markers.json").read())
        if lowercase == True:
            entity_marker_tokens = [t.lower() for t in entity_marker_tokens]
        tokenizer.add_tokens(entity_marker_tokens)

    # setup model
    if config["score_func"] == "biaffine":
        model = BiaffineNetwork(config)
    elif config["score_func"] == "dot":
        model = ConcatNonLinear(config)
    elif config["score_func"] == "cosine":
        model = ConcatNonLinearNormalized(config)
    elif config["score_func"] == "box":
        model = MinMaxBox(config)

    model.to(device)
    # create token embedding for new added markers
    model.encoder.resize_token_embeddings(len(tokenizer))

    # setup data
    time1 = time.time()

    data = Dataloader(config["data_path"], tokenizer, batchsize=config["train_batch_size"], shuffle=True,
                      max_text_length=config["max_text_length"], training=config["train"],
                      logger=logger, lowercase=lowercase, full_annotation=config["full_annotation"])
    logger.info(f"number of data points during training : {len(data)}")
    time2 = time.time()
    logger.info("Time spent loading data: %f" % (time2 - time1))

    config["max_num_steps"] = len(
        data) * config["epochs"] / config["train_batch_size"]
    if isinstance(config["log_interval"], float):
        config["log_interval"] = len(data) * config["log_interval"]
        logger.info(f"Log interval: {config['log_interval']}")
        print(len(data), config["log_interval"], config["log_interval"])
    if isinstance(config["warmup"], float):
        config["warmup"] = config["max_num_steps"] * config["warmup"]
        logger.info(f"warmup {config['warmup']}")

    return data, model, device, logger
