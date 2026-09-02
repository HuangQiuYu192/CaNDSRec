#!/usr/bin/env python3
import argparse
import os
import sys
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from argument_parser import build_config_dict, parse_args
from models import get_model_class


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})


def main():
    wrapper = argparse.ArgumentParser()
    wrapper.add_argument("--checkpoint", required=True)
    wrapper.add_argument("--force_cpu", action="store_true")
    known, remaining = wrapper.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model_class = get_model_class(args.model)
    config_dict = build_config_dict(args)
    if known.force_cpu:
        config_dict["use_gpu"] = False

    config = Config(model=model_class, dataset=args.dataset, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], args.model)(config, model)

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        model_file=known.checkpoint,
        show_progress=False,
    )
    logger.info("MODEL TEST FROM CHECKPOINT")
    logger.info("checkpoint: {}".format(known.checkpoint))
    logger.info("test result: {}".format(test_result))


if __name__ == "__main__":
    main()
