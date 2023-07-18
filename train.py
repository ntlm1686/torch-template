from utils import parse_config, get_run_name, get_dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import logging
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
import warnings
import modules
warnings.filterwarnings("ignore", ".*does not have many workers.*")

seed_everything(2, workers=True)

parser = argparse.ArgumentParser(description='Training the model')
parser.add_argument('config', type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--debug', '-d', action='store_true',
                    help='debug mode')
parser.add_argument('--wandb', '-w', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    # read config
    config = parse_config(args.config)

    # debug mode
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # wandb
    if args.wandb:
        wandb_logger = WandbLogger(
            project="Just a template",  # TODO: change this
            name=get_run_name(config),
        )

    trainset, valset = get_dataset(config)
    trainloader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    validloader = DataLoader(
        valset, batch_size=config.batch_size, num_workers=config.workers)

    lightning_model = modules.create(config, len(trainset))

    ckpt_callback = ModelCheckpoint(
        monitor='val_acc@1',
        dirpath='./ckpt',
        filename=get_run_name(config)+"/{epoch:02d}-{val_acc@1:.2f}-"+datetime.now().strftime('%m-%d-%H:%M'),
        every_n_epochs=1,
        save_weights_only=True,
        mode='max'
    )
    # Initialize a trainer
    trainer = Trainer(
        max_epochs=config.epoch,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            ckpt_callback,
            LearningRateMonitor(logging_interval='epoch')
        ],
        logger=wandb_logger if args.wandb else True,  # wandb logger
        precision=32,
        accelerator="gpu",
        devices=1,
        strategy="ddp_find_unused_parameters_false",   # suppress warning, same as `ddp`
        gradient_clip_val=0.5,
    )

    # Train the model ⚡
    trainer.fit(lightning_model, trainloader, validloader)
