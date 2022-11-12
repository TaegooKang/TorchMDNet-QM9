import os
import logging
import sys
import time
import yaml
from glob import glob
import torch

def warmup_schedule(step, warmup_steps, end_learning_rate=1.):
    initial_lr = 1 / warmup_steps
    step = min(step, warmup_steps)
    return ((initial_lr - end_learning_rate) * (1 - step / warmup_steps)) + end_learning_rate

def exponential_schedule(step, initial_lr, decay_rate, decay_steps):
    return initial_lr * decay_rate ** (step / decay_steps)

class LR_Scheduler():
    def __init__(self, initial_lr, warmup_steps, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        warmup_lr = warmup_schedule(step, self.warmup_steps, 1.)
        exponential_lr = exponential_schedule(step, self.initial_lr, self.decay_rate, self.decay_steps)
        
        return warmup_lr * exponential_lr
    
def save_args(args, save_path):
    with open(os.path.join(save_path,'hparams.yaml'), "w", encoding="utf8") as outfile:
        yaml.dump(args, outfile, default_flow_style=False, allow_unicode=True)

def save_checkpoint(net, epoch, save_path):
    print(' Saving the best model...')
    new_path = os.path.join(save_path, f"best_{epoch}.pt")

    for filename in glob.glob(os.path.join(save_path, "*.pt")):
        os.remove(filename)  # remove old checkpoint

    torch.save(net.state_dict(), new_path)
    
def set_logging_defaults(save_path):
    '''
    if os.path.isdir(logdir):
        res = input('"{}" exists. Overwrite [Y/n]? '.format(logdir))
        if res != 'Y':
            raise Exception('"{}" exists.'.format(logdir))
    else:
        os.makedirs(logdir)
    '''
    # set basic configuration for logging
    logging.basicConfig(format=" [%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(save_path, 'log.log')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    '''
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)
    '''

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates=99999):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]