import os, sys

path = os.path.split(__file__)[0]
# print("abs path is %s" %(os.path.abspath()))

config = {
    'batch_size' : 32,
    'val_batch_size':2,
    'num_iters':1000000000,
    'seed' : 1,
    'lr':3.16e-4,

    'lr_update_step':37500,
    'test_cycle':1,

    'cuda' : True,
    'gpus' :1,
    'gpuargs' : {'num_workers': 4, 
               'pin_memory' : True
              },

    'model':'ColorizationNet',
    'bachnorm':True,
    'pretrained':True,

    # 'opt_config':{
    #     'lr' : 0.001,
    #     'betas' : (0.9, 0.99),
    #     'eps': 1e-8,
    #     'weight_decay': 0.004
    # },

    'save' :'%s/work/' % path,

    'image_folder_train' : {
        'root' : '%s/' % path,
        'file' : '/media/chuchienshu/ACD26B99D26B6714/dataset/ILSVRC2012_img_train/*/*.JPEG',
        'replicates': 1,
        'train':True
    },
    'image_folder_val' : {
        'root' : '%s/' % path,
        'file' : '/home/chuchienshu/Downloads/dataset/DAVIS_test/*/*.jpg',
        'replicates': 1,
        'train':False
    },

    'log_frequency': 1, #frequency for the number of epotch
    'save_iamge':'%s/work/img/' % path
}

# print(config['save'])
