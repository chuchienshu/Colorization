# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
import os
import argparse

from datasets import  Image_from_folder
from config import config
from tensorboardX import SummaryWriter
from solver import Solver
# np.set_printoptions(threshold=np.nan)

def main(args):

    gpuargs = config['gpuargs'] if config['cuda'] else {}

    train_dataset = Image_from_folder(config['image_folder_train'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True ,**gpuargs, drop_last= True)

    val_dataset = Image_from_folder(config['image_folder_val'])
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False ,**gpuargs, drop_last= True)
        
    train_logger = SummaryWriter(log_dir = os.path.join(config['save'], 'train'), comment = 'training')
    val_logger = SummaryWriter(log_dir = os.path.join(config['save'], 'val'), comment = 'validation')

    solver = Solver(config)
    if args.train:

        solver.train(train_loader, val_loader, train_logger, val_logger,args.resume, args.valitate)
    if args.infer:

        solver.test(val_loader,val_logger,args.infer, save_model = False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--valitate','-V', type=bool, default=True, help='Decide whether to use cross validation')
    parser.add_argument('--train','-T', type=bool, default=True, help='Decide to train or not')
    # parser.add_argument('--pretrained','-P', type=bool, default=True, help='use pretrained model or not')
    parser.add_argument('--resume','-R', type=int, default=0, help='Specified resume time step')
    parser.add_argument('--infer','-I', type=int, default=0, help='Specified inference time step')
    args = parser.parse_args()
    print(args)
    main(args)
