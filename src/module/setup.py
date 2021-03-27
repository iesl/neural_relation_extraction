import time
from transformers import AutoTokenizer, AutoModel

from module.data_loader import Dataloader
from module.utils import cuda_if_available
from module.model import BRAN

__all__ = [
    "setup",
]

def setup(logger, config):
    """
    Setup and return the datasets, dataloaders, model, and training loop required for training.
    :param config: config dictionary
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """
    device = cuda_if_available(use_cuda=config["cuda"])

    # setup model
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_type"], use_fast=True)
    model = BRAN(config)
    model.to(device)

    # for name, param in model.encoder.named_parameters():
    #     if param.requires_grad:
    #         logger.debug('encoder', name, param.shape)
    # logger.debug('head_layer0' model.head_layer0.weight)
    # logger.debug('head_layer1', model.head_layer1.weight)
    # logger.debug('tail_layer0', model.tail_layer0.weight)
    # logger.debug('tail_layer1', model.tail_layer1.weight)
    # logger.debug('biaffine_mat', model.biaffine_mat)
    

    # setup data
    time1 = time.time()
    data = Dataloader(config["data_path"], tokenizer, batchsize=config["train_batch_size"], shuffle=True, 
                      seed=config["seed"], max_text_length=config["max_text_length"], training=config["train"],
                      logger=logger)
    logger.info(f"number of data points during training : {len(data)}")
    time2 = time.time()
    logger.info("Time spent loading data: %f" % (time2 - time1))

    if isinstance(config["log_interval"], float):
        config["log_interval"] = len(data) * config["log_interval"]
        logger.info(f"Log interval: {config['log_interval']}")

    return data, model, device
