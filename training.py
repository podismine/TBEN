import argparse
import os
import random
import shutil
import time
import warnings
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
import torch
import torch as t
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
from utils.visualize import Visualizer

from apex import amp
from apex.parallel import DistributedDataParallel
from models.TBEN import TBEN
import warnings 
from warmup_scheduler import GradualWarmupScheduler
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch TBEN Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 6400), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=0.001,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--env_name', default = "default", help='name for env')

args = parser.parse_args()
vis = Visualizer(args.env_name)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_sex, self.next_y, self.next_bc = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_sex = None
            self.next_y = None
            self.next_bc = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_sex = self.next_sex.cuda(non_blocking=True)
            self.next_y = self.next_y.cuda(non_blocking=True)
            self.next_bc = self.next_bc.cuda(non_blocking=True)


            self.next_input = self.next_input.float()
            self.next_target = self.next_target.float()
            self.next_sex = self.next_sex.float()
            self.next_y = self.next_y.float()
            self.next_bc = self.next_bc.float()
            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        sex = self.next_sex
        y = self.next_y
        bc = self.next_bc

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if sex is not None:
            sex.record_stream(torch.cuda.current_stream())
        if y is not None:
            y.record_stream(torch.cuda.current_stream())
        if bc is not None:
            bc.record_stream(torch.cuda.current_stream())

        self.preload()
        return input, target, sex, y, bc


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)

def loss_func(x, y):
        return torch.nn.MSELoss().cuda()(x, y) + torch.nn.L1Loss().cuda()(x, y)

def main_worker(local_rank, nprocs, args):
    best_acc1 = 99.0

    dist.init_process_group(backend='nccl')
    # create model

    model = TBEN(depth = 8, dim = 128, mlp_dim=512, pool = 'cls', tr_drop = 0.5)

    torch.cuda.set_device(local_rank)
    model.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)

    # define loss function (criterion) and optimizer

    criterion = None #loss_func


    optimizer = t.optim.Adam(model.parameters(),lr = args.lr,weight_decay = args.wd)
    model, optimizer = amp.initialize(model, optimizer)
    model = DistributedDataParallel(model)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = MultiBranch_Data("dataset_train.csv",train = True)
    val_data = MultiBranch_Data("dataset_valid.csv",train = False)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoader(train_data,args.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,args.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args)

        # evaluate on validation set
            
        mae= validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = mae < best_acc1 
        best_acc1 = min(mae, best_acc1)

        if not os.path.exists("checkpoints_s1/%s" % args.env_name):
            os.makedirs("checkpoints_s1/%s" % args.env_name)

        if is_best:
            if local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'best_acc1': best_acc1,
                        'amp': amp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, True ,'./checkpoints_s1/%s/%s_epoch_%s_%s' % (args.env_name, args.env_name, epoch, best_acc1))



def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae', ':6.2f')
    loss_kl = AverageMeter('kl', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, loss_mae,loss_kl],
                             prefix="Epoch: [{}]".format(epoch))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
    # switch to train mode
    

    model.train()
    end = time.time()
    prefetcher = data_prefetcher(train_loader)
    images, target, sex, y, bc= prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    while images is not None:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute output
        with autocast():
            out = model(images)
            prob = torch.exp(out)
            pred = torch.sum(prob * bc, dim = 1)

            loss = dpl.my_KLDivLoss(out, y) 
            mae = torch.nn.L1Loss()(pred, target) 


        loss_all = loss + 1.0 * mae
        torch.distributed.barrier() 

        reduced_loss = reduce_mean(loss_all, args.nprocs)
        reduced_mae = reduce_mean(mae, args.nprocs)
        reduced_kl = reduce_mean(loss, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        loss_mae.update(reduced_mae.item(), images.size(0))
        loss_kl.update(reduced_kl.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss_all, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        i += 1

        images, target, sex, y, bc= prefetcher.next()
    vis.plot('train_mae_loss', float(loss_mae.avg))
    vis.plot('train_kl_loss', float(loss_kl.avg))
    vis.plot('train_loss', float(losses.avg))

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mae = AverageMeter('mae1', ':6.2f')
    loss_kl = AverageMeter('kl', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, loss_mae, loss_kl], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        prefetcher = data_prefetcher(val_loader)
        images, target, sex, y, bc= prefetcher.next()
        i = 0
        while images is not None:

            # compute output
            with autocast():
                out= model(images)
                prob = torch.exp(out)
                pred = torch.sum(prob * bc, dim = 1)

                loss = dpl.my_KLDivLoss(out, y)
                mae1 = torch.nn.L1Loss()(pred, target) 


            torch.distributed.barrier()
            loss_all = loss + 1.0 * mae

            reduced_loss = reduce_mean(loss_all, args.nprocs)
            reduced_mae = reduce_mean(mae, args.nprocs)
            reduced_kl = reduce_mean(loss, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            loss_mae.update(reduced_mae.item(), images.size(0))
            loss_kl.update(reduced_kl.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            i += 1

            images, target, sex, y, bc= prefetcher.next()

        # TODO: this should also be done with the ProgressMeter
        print(' * MAE@ {loss_mae.avg:.3f} KL@ {loss_kl.avg:.3f}'.format(loss_mae1=loss_mae, loss_kl=loss_kl))
        vis.plot('val_mae_loss', float(loss_mae.avg))
        vis.plot('val_kl_loss', float(loss_kl.avg))
        vis.plot('val_loss', float(losses.avg))
    return loss_mae.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'final_model_best.pth.tar')
    


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr_warmup = [i / 200. * args.lr for i in range(1, 201)]
    if epoch < 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup[epoch]
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr #min(1e-6, (args.epochs - epoch) / args.epochs * args.lr)
    
if __name__ == '__main__':
    main()