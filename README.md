# Biomedical Relation Extraction with Document level distant supervision on CTD and PubMed

Create a new conda enviroment:

```
conda create -n <ENV_NAME> python=3.7
```

After cloning the repo, you can install all required dependencies by running the following in the root directory:

```
make install
```

Modify proper enviroment variables:

```
vim set_environment
```

Run the following command before training / testing:

```
source set_environment.sh
```

Then setup your wandb environment:

```
wandb init
```

An example of training on a slurm server:

```
srun --gres=gpu:1 --mem=25GB --partition=1080ti-long python src/main.py --data_path data/tacred/ --multi_class --partial_annotation --encoder_type bert-large-cased --score_func nn --learning_rate 1e-5 --max_text_length=128 --train_batch_size=2 --test_batch_size 16 --warmup 0.3 --dim=1024 --wandb
```
