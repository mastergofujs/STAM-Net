import logging
from pathlib import Path

import methods.feature as feature
import torch.optim as optim
from methods import ein_seld, triple_seld

from ruamel.yaml import YAML
from termcolor import cprint
from torch.utils.data import ConcatDataset, DataLoader

from utils.common import convert_ordinal, count_parameters, move_model_to_gpu
from utils.datasets import DCASE2020

method_dict = {
    'triple_seld': triple_seld,
}

datasets_dict = {
    'DCASE2020': DCASE2020,
}


def store_config(output_path, config):
    """ Write the given config parameter values to a YAML file.

    Args:
        output_path (str): Output file path.
        config: Parameter values to log.
    """
    yaml = YAML()
    with open(output_path, 'w') as f:
        yaml.dump(config, f)


# Datasets
def get_dataset(dataset_name, root_dir):
    assert dataset_name == 'DCASE2021' or dataset_name == 'DCASE2020', \
        "Dataset name '{}' is not 'DCASE2020' or 'DCASE2021'".format(dataset_name)
    dataset = datasets_dict[dataset_name](root_dir)
    print('\nDataset {} is being developed......\n'.format(dataset_name))
    return dataset


# Dataloaders
def get_generator(args, cfg, dataset, generator_type):
    """ Get generator.

    Args:
        args: input args
        cfg: configuration
        dataset: dataset used
        generator_type: 'train' | 'valid' | 'test'
            'train' for training, 'valid_train' for validation of train set,
            'valid' for validation of valid set.
    Output:
        subset: train_set or valid_set
        data_generator: 'train_generator', or 'valid_generator'
    """
    assert generator_type == 'train' or generator_type == 'valid' or generator_type == 'test', \
        "Data generator type '{}' is not 'train', 'valid_train' or 'valid'".format(generator_type)

    batch_sampler = None
    if generator_type == 'train':
       # len(subset)  1800
        subset = method_dict[cfg['method']].data.UserDataset(args, cfg, dataset, dataset_type='train')
        if 'pitchshift' in cfg['data_augmentation']['type']:
            augset = method_dict[cfg['method']].data.UserDataset(args, cfg, dataset, dataset_type='train_pitchshift')
            subset = ConcatDataset([subset, augset])
        
        batch_sampler = method_dict[cfg['method']].data.UserBatchSampler(
            clip_num=len(subset), 
            batch_size=cfg['training']['batch_size'], 
            seed=args.seed
        )
        data_generator = DataLoader(
            dataset=subset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=method_dict[cfg['method']].data.collate_fn,
            pin_memory=True
        )
    elif generator_type == 'valid':
        subset = method_dict[cfg['method']].data.UserDataset(args, cfg, dataset, dataset_type='valid')
        data_generator = DataLoader(
            dataset=subset,
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=method_dict[cfg['method']].data.collate_fn,
            pin_memory=True
        )
    elif generator_type == 'test':
        testset_type = cfg['inference']['testset_type']
        dataset_type = testset_type + '_test'
        subset = method_dict[cfg['method']].data.UserDataset(args, cfg, dataset, dataset_type=dataset_type)
        data_generator = DataLoader(
            dataset=subset,
            batch_size=cfg['inference']['batch_size'],
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=method_dict[cfg['method']].data.collate_fn_test,
            pin_memory=True
        )

    return subset, data_generator, batch_sampler


# Losses
def get_losses(cfg):
    """ Get losses

    """
    losses = method_dict[cfg['method']].losses.Losses(cfg)
    for idx, loss_name in enumerate(losses.names):
        logging.info('{} is used as the {} loss.'.format(loss_name, convert_ordinal(idx + 1)))
    logging.info('')
    return losses


# Metrics
def get_metrics(cfg, dataset):
    """ Get metrics

    """
    metrics = method_dict[cfg['method']].metrics.Metrics(dataset)
    for idx, metric_name in enumerate(metrics.names):
        logging.info('{} is used as the {} metric.'.format(metric_name, convert_ordinal(idx + 1)))
    logging.info('')
    return metrics


# Audio feature extractor
def get_afextractor(cfg, cuda):
    """ Get audio feature extractor

    """
    if cfg['data']['audio_feature'] == 'logmel&intensity':
        afextractor = feature.LogmelIntensity_Extractor(cfg)
    afextractor = move_model_to_gpu(afextractor, cuda)
    return afextractor


# Models
def get_models(cfg, dataset, cuda, model_name=None):
    """ Get models

    """
    logging.info('=====>> Building a model\n')
    if not model_name:
        # TODO: dual_former
         model = vars(method_dict[cfg['method']])[cfg['training']['model']](
            cfg, dataset, encoder_kwargs=cfg["model"]["encoder"])
        
        # TODO: triple_former
        #model = vars(method_dict[cfg['method']])[cfg['training']['model']](
        #                cnn_kwargs=cfg["model"]["cnn"],
        #                encoder_kwargs=cfg["model"]["encoder"])
    else:
        # TODO: dual_former
        model = vars(method_dict[cfg['method']])[cfg['training']['model']](
            cfg, dataset, encoder_kwargs=cfg["model"]["encoder"])
        # TODO: triple_former
        #model = vars(method_dict[cfg['method']])[cfg['training']['model']](
        #    cnn_kwargs=cfg["model"]["cnn"],
        #    encoder_kwargs=cfg["model"]["encoder"])
        
    model = move_model_to_gpu(model, cuda)
    logging.info('Model architectures:\n{}\n'.format(model))
    count_parameters(model)
    return model


# Optimizers
def get_optimizer(cfg, af_extractor, model):
    """ Get optimizers
    
    """
    opt_method = cfg['training']['optimizer']
    lr = cfg['training']['lr']
    params = list(af_extractor.parameters()) + list(model.parameters())
    if opt_method == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    elif opt_method == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0, weight_decay=0)
    elif opt_method == 'adamw':
        # optimizer = AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0, warmup=0)
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=0.01, amsgrad=True)

    logging.info('Optimizer is: {}\n'.format(opt_method))
    return optimizer

# Trainer
def get_trainer(args, cfg, dataset, valid_set, af_extractor, model, optimizer, losses, metrics):
    """ Get trainer

    """
    trainer = method_dict[cfg['method']].training.Trainer(
        args=args, cfg=cfg, dataset=dataset, valid_set=valid_set, af_extractor=af_extractor,
        model=model, optimizer=optimizer, losses=losses, metrics=metrics
    )
    return trainer

# Inferer
def get_inferer(cfg, dataset, af_extractor, model, cuda):
    """ Get inferer

    """
    inferer = method_dict[cfg['method']].inference.Inferer(
        cfg=cfg, dataset=dataset, af_extractor=af_extractor, model=model, cuda=cuda
    )
    return inferer
