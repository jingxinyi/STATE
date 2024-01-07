import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer_flow_3d = config.init_obj('optimizer_flow_3d', torch.optim,
                                        filter(lambda p: p.requires_grad, model.FlowModule3D.parameters()))
    optimizer_flow_disc = config.init_obj('optimizer_disc', torch.optim,
                                          filter(lambda p: p.requires_grad, model.DiscriminatorModule.parameters()))

    optimizer = {"flow_3d": optimizer_flow_3d,
                 "disc": optimizer_flow_disc}

    lr_scheduler = {k: config.init_obj('lr_scheduler', torch.optim.lr_scheduler, v) for k, v in optimizer.items()}

    trainer = Trainer(model, criterion, metrics,
                      optimizer=optimizer,
                      config=config,
                      data_loader=data_loader,
                      gan_start_step=config['trainer'].get('gan_start_step'),
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      len_epoch=config['trainer'].get('len_epoch', None))

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Novel View Synthesis')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
