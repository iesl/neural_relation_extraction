from module.setup import setup
from module.train import Trainer
import click
import json
import sys
import os
os.environ['TRANSFORMERS_CACHE'] = os.environ['BIORE_OUTPUT_ROOT'] + \
    '/.cache/huggingface/'


class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


@click.command(
    context_settings=dict(show_default=True),
)
@click.option(
    "--train / --test",
    default=True,
    help="enable/disable training",
)
@click.option(
    "--data_path",
    type=click.Path(),
    default="data/",
    help="directory or file",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default="",
    help="directory to load / save model",
)
@click.option(
    "--encoder_type",
    type=click.Choice(
        ["bert-base-cased", "bert-base-uncased", "bert-large-cased", "bert-large-uncased",
         "roberta-large", "roberta-base", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"],
        case_sensitive=False,
    ),
    default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    help="encoder architecture to use",
)
@click.option(
    "--score_func",
    type=click.Choice(
        ["biaffine", "nn", "box"],
        case_sensitive=False,
    ),
    default="biaffine",
    help="aggregation layer of multiple mentions",
)
@click.option(
    "--multi_label / --multi_class",
    default=True,
    help="multi_label allows multiple labels during inference; multi_class only allow one label"
)
@click.option(
    "--na_mode",
    type=click.Choice(
        ["0", "1", "2", "3", "4"],
        case_sensitive=False,
    ),
    default="0",
    help="na_mode"
    + "0: na as a vector / box"
    + "1: has a binary classifier on top for no_relation detection, and then cascade another layer of softmax"
    + "2: hardcoded constraint that type box should be inside not_na box (will be equivalent to 2 for nn / biaffine)"
    + "3: regularizing type boxes to be inside not_na box (will be equivalent to 2 for nn / biaffine)"
    + "4: 2 + 3 (will be equivalent to 2 for nn / biaffine)"
    + "(ignored when --multi_label is set)"
)
@click.option(
    "--categorical_weight",
    type=float,
    default=0.01,
    help="weight of categorical classifier (w/o NA) loss term, ignore if na_mode == 0"
)
@click.option(
    "--na_weight",
    type=float,
    default=0.01,
    help="weight of binary classifier (exist a relation or not) loss term, ignore if na_mode == 0"
)
@click.option(
    "--na_box_weight",
    type=float,
    default=0.01,
    help="weight for let not_na box to contain other relation boxes, ignored if score_func != box or na_mode not 3, 4"
)
@click.option(
    "--full_annotation / --partial_annotation",
    default=True,
    help="full_annotation means all unlabeled pairs indicate no relation; "
    + "partial_annotation means we don't train or evaluate on unlabeled pairs."
)
@click.option(
    "--train_batch_size",
    type=int,
    default=2,
    help="batch size for training",
)
@click.option(
    "--grad_accumulation_steps",
    type=int,
    default=16,
    help="tricks to have larger batch size with limited memory."
    + " The real batch size = train_batch_size * grad_accumulation_steps",
)
@click.option(
    "--test_batch_size",
    type=int,
    default=16,
    help="batch size for training",
)
@click.option(
    "--max_text_length",
    type=int,
    default=512,
    help="max doc length of BPE tokens",
)
@click.option(
    "--dim",
    type=int,
    default=512,
    help="dimension of last layer feature before the score function "
    + "(e.g., dimension of biaffine layer, dimension of boxes)",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="learning rate",
)
@click.option(
    "--weight_decay",
    type=float,
    default=0.0,
    help="weight decay",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=0.1,
    help="dropout rate",
)
@click.option(
    "--volume_temp",
    type=float,
    default=1.0,
    help="volume temp for box (default 1.0)",
)
@click.option(
    "--intersection_temp",
    type=float,
    default=1e-3,
    help="intersection temp for box (default 1e-3)",
)
@click.option(
    "--max_grad_norm",
    type=float,
    default=1.0,
    help="gradient norm clip (default 1.0)",
)
@click.option("--epochs", type=int, default=5, help="number of epochs to train")
@click.option(
    "--patience",
    type=int,
    default=5,
    help="patience parameter for early stopping",
)
@click.option(
    "--log_interval",
    type=IntOrPercent(),
    default=0.25,
    help="interval or percentage (as float in [0,1]) of examples to train before logging training metrics "
    "(default: 0, i.e. every batch)",
)
@click.option(
    "--warmup",
    type=float,
    default=-1.0,
    help="number of examples or percentage of training examples for warm up training "
    "(default: -1.0, no warmup, constant learning rate",
)
@click.option(
    "--lambda",
    type=float,
    default=2.0,
    help="",
)
@click.option(
    "--m_pos",
    type=float,
    default=2.5,
    help="",
)
@click.option(
    "--m_neg",
    type=float,
    default=0.5,
    help="",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="seed for random number generator",
)
@click.option(
    "--cuda / --no_cuda",
    default=True,
    help="enable/disable CUDA (eg. no nVidia GPU)",
)
@click.option(
    "--wandb / --no_wandb",
    default=False,
    help="enable/disable wandb",
)
@click.option(
    "--save_embedding / --not_save_embedding",
    default=False,
    help="enable / disable to save sentence embeddings for dev set and relation matrix",
)
def main(**config):

    if config["output_path"] != "" and config["train"] == False:
        config_path = os.path.join(config["output_path"], 'log')
        config_str = open(config_path).read().split('\n')[1].split(
            ' - ')[1].replace("'", '"').replace("True", "true").replace("False", "false")
        config_load = json.loads(config_str)
        config_load["train"] = config["train"]
        config_load["wandb"] = config["wandb"]
        config_load["output_path"] = config["output_path"]
        config_load["cuda"] = config["cuda"]
        for key in config:
            if key not in config_load:
                config_load[key] = config[key]
        config = config_load

    data, model, device, logger = setup(config)
    trainer = Trainer(data, model, logger, config, device)
    if config["train"] == True:
        trainer.train()
    else:
        best_metric_threshold = trainer.load_model()
        trainer.model.eval()
        macro_perf, micro_perf, categ_acc, categ_macro_perf, na_acc, not_na_perf, na_perf, per_rel_perf = trainer.test(
            test=True, best_metric_threshold=best_metric_threshold)
        trainer.local_logging(micro_perf, macro_perf, categ_acc, categ_macro_perf,
                              na_acc, not_na_perf, na_perf, per_rel_perf, 0, label="TEST")

    logger.info("Program finished")


if __name__ == "__main__":
    main()
