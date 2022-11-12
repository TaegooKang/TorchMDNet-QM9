import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch_geometric.seed import seed_everything

from dataset import get_qm9_loader
from model import build_torchmdnet
from utils import AverageMeter, LR_Scheduler, EMA, set_logging_defaults, save_args, progress_bar, save_checkpoint 


# get args
parser = argparse.ArgumentParser()
parser.add_argument('--total-steps', type=int, default=3000000)
parser.add_argument('--batch-size', type=int, default=128, help='mini-batch size')
parser.add_argument('--initial-lr', type=float, default=4e-4)
parser.add_argument('--warmup-steps', type=int, default=10000)
parser.add_argument('--decay-rate', type=float, default=0.1)
parser.add_argument('--decay-steps', type=int, default=400000) 
parser.add_argument('--weight-decay', type=float, default=0.) # 0이 제일 좋음
parser.add_argument('--save-path', type=str, default='results/')
parser.add_argument('--optim', type=str, default='adamw')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--ema-decay', type=float, default=1.)
args = parser.parse_args()

# mask save dir
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

# set seed
seed = 1234
seed_everything(seed)

# save arguments
save_args(args, args.save_path)

# set logger
set_logging_defaults(args.save_path)

# build model
net = build_torchmdnet()
net = net.to(device)

# set ema(exponential moving average)
ema = EMA(net, decay=args.ema_decay)

# set loss function
if args.loss == 'mse':
    criterion = nn.MSELoss()
elif args.loss == 'l1':
    criterion = nn.L1Loss()
elif args.loss == 'smooth_l1':
    criterion = nn.SmoothL1Loss(beta=0.1)

# set optimizer and lr scheduler
if args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay, amsgrad=True)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)    
lr_scheduler = LR_Scheduler(args.initial_lr, args.warmup_steps, args.decay_rate, args.decay_steps)

# get train, valid loader
train_loader, valid_loader = get_qm9_loader('train', batch_size=args.batch_size)     

# set training config
total_steps = args.total_steps
cur_step = 1
cur_epoch = 1
training = True
best_mae = 1000000

# train <args.total_steps> iteration
while training:
    train_loss = AverageMeter('train loss')
    train_mae = AverageMeter('train mae')
    valid_loss = AverageMeter('valid loss')
    valid_mae = AverageMeter('valid mae')
    
    # train epoch
    net.train()
    for step, batch in enumerate(train_loader):
        lr = lr_scheduler(cur_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        pred, _ = net(batch.z.to(device), batch.pos.to(device), batch.batch.to(device))
        pred = pred.squeeze()
        
        loss = criterion(pred, batch.y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mae = torch.mean(torch.abs(pred-batch.y.to(device)))
        train_loss.update(loss.item())
        train_mae.update(mae.item())
        
        msg = f'train loss: {train_loss.avg:.4f} | train mae: {train_mae.avg:.6f} | lr: {lr:.8f}'
        progress_bar(step, len(train_loader), msg)
        
        ema(net)
        cur_step += 1
        if cur_step > total_steps:
            training = False
            break
    
    logger = logging.getLogger('train')
    logger.info(f'[Epoch {cur_epoch}] [Loss {train_loss.avg:.4f}] [MAE {train_mae.avg:.6f}] [LR {lr:.8f}]')  
    
    # validation
    net.eval()
    ema.assign(net)
    for step, batch in enumerate(valid_loader):
        pred, _ = net(batch.z.to(device), batch.pos.to(device), batch.batch.to(device))
        pred = pred.squeeze()
        loss = criterion(pred, batch.y.to(device))
        mae = torch.mean(torch.abs(pred-batch.y.to(device)))
        valid_loss.update(loss.item())
        valid_mae.update(mae.item())
        
        msg = f'valid loss: {valid_loss.avg:.4f} | valid mae: {valid_mae.avg:.6f}'
        progress_bar(step, len(valid_loader), msg)
    
    logger = logging.getLogger('valid')
    logger.info(f'[Epoch {cur_epoch}] [Loss {valid_loss.avg:.4f}] [MAE {valid_mae.avg:.6f}]')  
    if valid_mae.avg < best_mae:
        best_mae = valid_mae.avg
        save_checkpoint(net, cur_epoch, args.save_path)
            
    cur_epoch += 1
    ema.resume(net)
# save last model
torch.save(net.state_dict(), os.path.join(args.save_path, 'last.pt'))



