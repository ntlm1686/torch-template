# extract the model from pytorch-lightning checkpoint

import torch
from modules import DefaultModule

import argparse
parser = argparse.ArgumentParser(description='Extract weight from lightning ckpt')
parser.add_argument('--ckpt', type=str, help='checkpoint path, XFormer-N16 by default')
args = parser.parse_args()


lm = DefaultModule.load_from_checkpoint(args.ckpt, strict=False)
xformer = lm.model
xformer.eval()

torch.save(xformer, "../pi/xformer.pt")

print("Model saved to pi/xformer.pt")
