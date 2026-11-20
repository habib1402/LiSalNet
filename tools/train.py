#!/usr/bin/env python3
"""Single-device LiSalNet training with the Section 3 deep-supervision loss.

Each epoch has three complete passes: 256, 288, then the configured main size
(320 by default). Augmentation, Adam settings and polynomial LR follow the
uploaded implementation. A held-out validation list is optional; no benchmark
test split is automatically used to select weights.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import LiSalNet
from dataset import Dataset
import Transforms as T
from losses import DeepSupervisionLoss, BCEDiceLoss
from checkpoint_utils import read_checkpoint, load_model_state, load_checkpoint
from SalEval import SalEval

ARCHITECTURE = 'LiSalNet_without_RGB_block_18'


def transforms(width, height, training):
    operations = [T.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]), T.Scale(width, height)]
    if training:
        operations += [T.RandomCropResize(int(7. / 224. * width)), T.RandomFlip()]
    return T.Compose(operations + [T.ToTensor()])


def adjust_learning_rate(args, optimizer, epoch, step, max_batches):
    lr = args.lr * (1 - step / (max_batches * args.max_epochs)) ** 0.9
    if epoch == 0 and step < 200:
        lr = args.lr * 0.9 * (step + 1) / 200 + 0.1 * args.lr
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def train_pass(model, loader, criterion, optimizer, args, device, epoch, step, max_batches):
    model.train()
    total = 0.0
    for rgb, mask, depth in loader:
        lr = adjust_learning_rate(args, optimizer, epoch, step, max_batches)
        rgb, mask, depth = rgb.to(device), mask.float().to(device), depth.float().to(device)
        prediction = model(rgb, depth)
        loss = criterion(prediction, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        step += 1
    return total / len(loader), step, lr


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metric = SalEval(device=device)
    total = 0.0
    for rgb, mask, depth in loader:
        rgb, mask, depth = rgb.to(device), mask.float().to(device), depth.float().to(device)
        prediction = model(rgb, depth)
        total += criterion(prediction, mask).item()
        metric.addBatch(prediction[:, 0], mask)
    f_beta, mae = metric.getMetric()
    return {'loss': total / len(loader), 'F_beta': f_beta, 'MAE': mae}


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file_root', required=True)
    parser.add_argument('--train_list', default='NJU2K_NLPR_train')
    parser.add_argument('--val_list', default=None, help='Held-out training validation manifest, without .txt.')
    parser.add_argument('--depth_postfix', default='_depth.png')
    parser.add_argument('--inWidth', type=int, default=320)
    parser.add_argument('--inHeight', type=int, default=320)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--savedir', default='./checkpoints/LiSalNet_updated')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--onGPU', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--no_imagenet', action='store_true', help='Initialize retained RGB weights randomly.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Full training checkpoint from this revised package only.')
    group.add_argument('--weights', help='Original or revised checkpoint for model-weight initialization only.')
    args = parser.parse_args()
    if min(args.inWidth, args.inHeight, args.max_epochs, args.batch_size) <= 0:
        parser.error('Image sizes, epochs and batch size must be positive.')
    if args.onGPU and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --onGPU False to run on CPU.')
    device = torch.device('cuda:0' if args.onGPU else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.onGPU:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # Retained training backend policy.

    savedir = Path(args.savedir)
    if savedir.exists() and any(savedir.iterdir()) and not args.resume:
        parser.error('Output directory is not empty; choose a new directory or use --resume.')
    savedir.mkdir(parents=True, exist_ok=True)
    model = LiSalNet(pretrained=not (args.no_imagenet or args.weights or args.resume)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    criterion = DeepSupervisionLoss()
    if args.weights:
        load_checkpoint(model, args.weights)
    loaders = []
    for index, (width, height) in enumerate([(256, 256), (288, 288), (args.inWidth, args.inHeight)]):
        dataset = Dataset(args.train_list, args.file_root, transforms(width, height, True), True, args.depth_postfix)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=index < 2, drop_last=True))
    if any(len(loader) == 0 for loader in loaders):
        parser.error('Training set must contain at least one full batch.')
    val_loader = None
    if args.val_list:
        data = Dataset(args.val_list, args.file_root, transforms(args.inWidth, args.inHeight, False), True, args.depth_postfix)
        if set(data.images) & set(loaders[0].dataset.images):
            parser.error('Training and validation image lists overlap.')
        if 'test' in args.val_list.lower():
            parser.error('Use a held-out training validation split, not a test-named manifest.')
        val_loader = DataLoader(data, batch_size=10, num_workers=args.num_workers, shuffle=False)
    max_batches = sum(len(loader) for loader in loaders)
    start_epoch, step, best_f, best_mae = 0, 0, -1.0, float('inf')
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        required = {'architecture', 'optimizer', 'epoch', 'global_step', 'rng', 'config'}
        if not required.issubset(checkpoint) or checkpoint['architecture'] != ARCHITECTURE:
            parser.error('Legacy weights cannot resume Adam state; use --weights for initialization.')
        for key in ['max_epochs', 'batch_size', 'lr', 'inWidth', 'inHeight', 'train_list', 'val_list', 'file_root', 'depth_postfix', 'num_workers']:
            if checkpoint['config'].get(key) != getattr(args, key):
                parser.error(f'Resume configuration differs for {key}.')
        removed = load_model_state(model, checkpoint)
        if removed:
            parser.error('Checkpoint contains old block 18; use --weights instead.')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, step = checkpoint['epoch'], checkpoint['global_step']
        if step != start_epoch * max_batches:
            parser.error('Resume loader length differs from the checkpoint training schedule.')
        best_f, best_mae = checkpoint['best_f_beta'], checkpoint['best_mae']
        restore_rng(checkpoint['rng'])
    (savedir / 'configuration.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    print('Parameters:', sum(p.numel() for p in model.parameters()), 'Batches per epoch:', max_batches)
    for epoch in range(start_epoch, args.max_epochs):
        losses = []
        for loader in loaders:
            loss, step, lr = train_pass(model, loader, criterion, optimizer, args, device, epoch, step, max_batches)
            losses.append(loss)
        val_result = validate(model, val_loader, criterion, device) if val_loader is not None else None
        is_best = val_result is not None and (val_result['F_beta'] > best_f or
                  (val_result['F_beta'] == best_f and val_result['MAE'] < best_mae))
        if is_best:
            best_f, best_mae = val_result['F_beta'], val_result['MAE']
        record = {'epoch': epoch + 1, 'global_step': step, 'losses_256_288_main': losses, 'lr': lr, 'validation': val_result}
        print(json.dumps(record))
        with (savedir / 'training.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        checkpoint = {'architecture': ARCHITECTURE, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'global_step': step,
                      'best_f_beta': best_f, 'best_mae': best_mae, 'config': vars(args), 'rng': rng_state()}
        temporary = savedir / 'checkpoint.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(savedir / 'checkpoint.pth.tar')
        if is_best:
            torch.save(model.state_dict(), savedir / 'best_model.pth')
    torch.save(model.state_dict(), savedir / 'model_final.pth')


if __name__ == '__main__':
    main()
