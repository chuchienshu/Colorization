# freda (todo) : 

import os, time, sys, math
import subprocess, shutil
from os.path import *
import numpy as np
from inspect import isclass
from pytz import timezone
from datetime import datetime
import inspect
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import  yuv2rgb, lab2rgb
from skimage import io
# import cv2

def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)

def module_to_dict(module, exclude=[]):
        return dict([(x, getattr(module, x)) for x in dir(module)
                     if isclass(getattr(module, x))
                     and x not in exclude
                     and getattr(module, x) not in exclude])

class TimerBlock: 
    def __init__(self, title):
        print("{}".format(title))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")


    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print("  [{:.3f}{}] {}".format(duration, units, string))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def add_arguments_for_module(parser, module, argument_for_class, default, skip_params=[], parameter_defaults={}):
    argument_group = parser.add_argument_group(argument_for_class.capitalize())

    module_dict = module_to_dict(module)
    argument_group.add_argument('--' + argument_for_class, type=str, default=default, choices=module_dict.keys())
    
    args, unknown_args = parser.parse_known_args()
    class_obj = module_dict[vars(args)[argument_for_class]]

    argspec = inspect.getargspec(class_obj.__init__)

    defaults = argspec.defaults[::-1] if argspec.defaults else None

    args = argspec.args[::-1]
    for i, arg in enumerate(args):
        cmd_arg = '{}_{}'.format(argument_for_class, arg)
        if arg not in skip_params + ['self', 'args']:
            if arg in parameter_defaults.keys():
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(parameter_defaults[arg]), default=parameter_defaults[arg])
            elif (defaults is not None and i < len(defaults)):
                argument_group.add_argument('--{}'.format(cmd_arg), type=type(defaults[i]), default=defaults[i])
            else:
                print("[Warning]: non-default argument '{}' detected on class '{}'. This argument cannot be modified via the command line"
                        .format(arg, module.__class__.__name__))
            # We don't have a good way of dealing with inferring the type of the argument
            # TODO: try creating a custom action and using ast's infer type?
            # else:
            #     argument_group.add_argument('--{}'.format(cmd_arg), required=True)

def kwargs_from_args(args, argument_for_class):
    argument_for_class = argument_for_class + '_'
    return {key[len(argument_for_class):]: value for key, value in vars(args).items() if argument_for_class in key and key != argument_for_class + 'class'}

def format_dictionary_of_losses(labels, values):
    try:
        string = ', '.join([('{}: {:' + ('.3f' if value >= 0.001 else '.1e') +'}').format(name, value) for name, value in zip(labels, values)])
    except (TypeError, ValueError) as e:
        print(zip(labels, values))
        string = '[Log Error] ' + str(e)

    return string


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.next()
        self.last_duration = (time.time() - start)
        return n

    next = __next__

def gpumemusage():
    gpu_mem = subprocess.check_output("nvidia-smi | grep MiB | cut -f 3 -d '|'", shell=True).replace(' ', '').replace('\n', '').replace('i', '')
    all_stat = [float(a) for a in gpu_mem.replace('/','').split('MB')[:-1]]

    gpu_mem = ''
    for i in range(len(all_stat)/2):
        curr, tot = all_stat[2*i], all_stat[2*i+1]
        util = "%1.2f"%(100*curr/tot)+'%'
        cmem = str(int(math.ceil(curr/1024.)))+'GB'
        gmem = str(int(math.ceil(tot/1024.)))+'GB'
        gpu_mem += util + '--' + join(cmem, gmem) + ' '
    return gpu_mem


def update_hyperparameter_schedule( epoch, global_iteration, optimizer):
    boundaries = [400000, 600000, 800000 ,1000000, 1200000]
    lrs = [1e-4,0.5e-4, 0.25e-4, 0.125e-4, 0.625e-5]
    boundaries.append(global_iteration)
    boundaries.sort()
    lr = lrs[boundaries.index(global_iteration)]   
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, path, prefix,epoch, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    # name = prefix_save + '_' + filename
    name = '%s_%d_%s' % (prefix_save,epoch,filename)
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def img_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 255)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated
    plt.pause(200)  # pause a bit so that plots are updated
    


def save_imgs(tensor, filename):

    for index, im in enumerate(tensor):
        # print(im.shape)
        # im =np.clip(im.numpy().transpose(1,2,0), -1, 1) 
        img_rgb_out = (255*np.clip(lab2rgb(im),0,1)).astype('uint8')
        io.imsave(filename +'rgb'+ str(index) + '.png', img_rgb_out )

def get_allocated(filename = 'objs'):
    import gc
    obj_file = open(filename, 'a')
    obj_file.write('#####################' + '\n')
    total_size = 0

    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            nums =  obj.nelement() * 4
            print(type(obj), obj.size(), nums)
            
            # obj_file.write(str(type(obj)) + str(obj.size()) + '\n')
            total_size += nums

    obj_file.write('total_size ------> %s MB' % str(total_size >> 16) + '\n')
    obj_file.flush()
    obj_file.close()

def resize_flow(flow, size = None):
    '''
    resize the flow to special size, and nomornalize it's value to [-1, 1]
    para:
         flow is a numpy array with shape [h, w, 2]
         size is a list obj like [target_h, target_w]
    return:
        resized and normalized flow-filed.
    '''
    assert isinstance(flow, np.ndarray)
    factor = np.max(np.abs(flow) )
    flow /= factor
    if size is None:
        return flow * factor
    return resize(flow , size) * factor

def warp_forward_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


import random
import torch
from torch.autograd import Variable


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images