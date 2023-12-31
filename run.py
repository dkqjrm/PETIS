import torch
from model import Classifier
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, ModelSummary, RichModelSummary
import numpy as np
import random
import argparse


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default='crawled_all_dataset_0613_v4.jsonl', type=str)
    parser.add_argument("--model_name", default='klue/roberta-large', type=str)

    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--pre_seq_len', type=int, default=50)
    parser.add_argument('--prefix_projection', type=bool, default=True)
    parser.add_argument('--prefix_hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=24)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument("--epoch", default=100, type=int)

    parser.add_argument('--devices', nargs='+', type=int, default=[1], help='list of device ids')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    args = vars(parser.parse_args())

    fix_seed(args["seed"])

    model = Classifier(config=args)

    logger = TensorBoardLogger(save_dir="./check",
                               name="lr{}_batch{}_epoch{}_pat{}_ac{}_prelen{}_wd{}_pp{}".format(args["learning_rate"],
                                                                                                args["batch_size"],
                                                                                                args['epoch'],
                                                                                                args['patience'],
                                                                                                args["accumulate"],
                                                                                                args['pre_seq_len'],
                                                                                                args['weight_decay'],
                                                                                                args[
                                                                                                    'prefix_projection']))

    logger.log_hyperparams(args)

    early_stopping = EarlyStopping("train/loss_epoch", patience=args["patience"], mode='min')

    checkpoint = ModelCheckpoint(dirpath="./output/{}".format(args["seed"]),
                                 filename="{epoch}_{train/loss_epoch:.2f}_" + "lr{}_batch{}_prelen{}".format(
                                     args["learning_rate"], args["batch_size"], args['pre_seq_len']),
                                 monitor="train/loss_epoch",
                                 mode="max",
                                 save_top_k=1,
                                 save_last=True
                                 )

    trainer = Trainer(accelerator="gpu",
                      devices=args["devices"],
                      max_epochs=args["epoch"],
                      accumulate_grad_batches=args['accumulate'],
                      logger=logger,
                      callbacks=[early_stopping, checkpoint, ModelSummary()],
                      log_every_n_steps=1, )

    trainer.fit(model)
    trainer.test(model, ckpt_path='best')
    loaded_model = Classifier.load_from_checkpoint(checkpoint.best_model_path)
    trainer.test(loaded_model)
