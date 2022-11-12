import argparse, os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset import get_qm9_loader
from model import build_torchmdnet
from utils import progress_bar

# get args
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='path for network weights', default='best_result/best_588.pt')
args = parser.parse_args()

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

# build model
net = build_torchmdnet()
net.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
net = net.to(device)

# set test loader
test_loader = get_qm9_loader('test', batch_size=32)

# test
net.eval()
preds = []
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        pred,_ = net(batch.z.to(device), batch.pos.to(device), batch.batch.to(device))
        preds.append(pred.cpu().detach().numpy())
        progress_bar(step, len(test_loader), 'Testing...')
preds = np.concatenate(preds, axis=0)
np.savetxt(os.path.join(args.checkpoint.split('/')[0],'pred3.csv'), preds)