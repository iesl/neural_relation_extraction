import os
os.environ['TRANSFORMERS_CACHE']=os.environ['PYTHONPATH'] + '/.cache/huggingface/'

import sys
import click
from module import logger
from module.train import Trainer
from module.setup import setup

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
        ["bert-base-cased", "bert-base-uncased", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"],
        case_sensitive=False,
    ),
    default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    help="encoder architecture to use",
)
@click.option(
    "--aggregation",
    type=click.Choice(
        ["lse"],
        case_sensitive=False,
    ),
    default="lse",
    help="aggregation layer of multiple mentions",
)
@click.option(
    "--train_batch_size",
    type=int,
    default=8,
    help="batch size for training",
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
    help="dimension for hidden embedding",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="learning rate",
)
@click.option("--epochs", type=int, default=20, help="number of epochs to train")
@click.option(
    "--patience",
    type=int,
    default=3,
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
    "--cuda / --no_cuda",
    default=True,
    help="enable/disable CUDA (eg. no nVidia GPU)",
)
@click.option(
    "--wandb / --no_wandb",
    default=True,
    help="enable/disable wandb",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="seed for random number generator",
)
def main(**config):
    logger.info("Program started")
    logger.info(config)
    data, model, device = setup(logger, config)
    trainer = Trainer(data, model, logger, config, device)
    if config["train"] == True:
        trainer.train()
    else:
        best_metric_threshold = trainer.load_model()
        macro_perf, micro_perf, per_rel_perf = trainer.test(test=True, best_metric_threshold=best_metric_threshold)
        logger.info(f"Test: Macro Precision={macro_perf['P']}, Macro Recall={macro_perf['R']}, Macro F1={macro_perf['F']}")
        logger.info(f"Test: Micro Precision={micro_perf['P']}, Micro Recall={micro_perf['R']}, Micro F1={micro_perf['F']}")
        for rel_name, (pp, rr, ff, tt) in per_rel_perf.items():
            logger.info(f"TEST: {rel_name}, Precision={pp}, Recall={rr}, F1={ff}, threshold={tt}")
    logger.info("Program finished")
    

if __name__ == "__main__":
    main()
