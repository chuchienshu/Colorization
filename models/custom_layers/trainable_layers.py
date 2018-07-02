# **************************************
# ***** Richard Zhang / 2016.08.06 *****
# **************************************
import numpy as np
import warnings
import os
import sklearn.neighbors as neighbors
import torch
from skimage import color
from torch.autograd import Function
# ************************
# ***** CAFFE LAYERS *****
# ************************

class NNEncode():
    ''' Encode points using NearestNeighbors search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = neighbors.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False,sameBlock=True):
        '''
        将输入的feature map flatten 成 [N*H*W, 2]，然后在 313 个 bin 中找到与其自身 ab 值最近的 NN(此处取 10) 个’调色板‘的 ab 值，并将对应的距离值作数学处理后返回到新的 feature map [N, 313, H, W]的对应位置并返回。
        '''
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        #pts_flt ---> [N*H*W, 2]
        P = pts_flt.shape[0]
        #P ---> N*H*W
        if(sameBlock and self.alreadyUsed):
            #避免反复分配全 0 数组浪费时间
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            #self.pts_enc_flt.shape ---> [N*H*W, 313]
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]
            #self.p_inds.shape ---> [N*H*W, 1]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)
        #inds.shape ---> [N*H*W, NN]

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]
        #wts.shape ---> [N*H*W, NN]
        
        #将输入的 feature map(ab 值)与调色板 bin 中最近的 NN(此处取 10) 个距离值赋值到 pts_enc_flt 中，然后展开成 4d 形式返回。
        self.pts_enc_flt[self.p_inds,inds] = wts
        #shape mismatch: indexing arrays could not be broadcast together with shapes (16384,1) (4096,10) 
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)
        #pts_enc_nd.shape  -----> [N, 313, H, W]

        return pts_enc_nd.astype('float32')

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd


class NNEncLayer(object):
    ''' Layer which encodes ab map into Q colors,ab_enc
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def __init__(self):
        warnings.filterwarnings("ignore")

        self.NN = 1
        self.sigma = 5.
        self.ENC_DIR = './models/custom_layers/'
        self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

    def __call__(self,bottom):
        if len(bottom) == 0:
            raise Exception("NNEncLayer should have inputs")
        return self.nnenc.encode_points_mtx_nd(bottom,axis=1)

class PriorBoostLayer(object):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def __init__(self ):

        self.ENC_DIR = './models/custom_layers/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

    def __call__(self, bottom):
        if len(bottom) == 0:
            raise Exception("PriorBoostLayer should have inputs")
        return self.pc.forward(bottom ,axis=1)

class NonGrayMaskLayer(object):
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS    
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''

    def __init__(self):
        self.thresh = 5 # threshold on ab value

    def __call__(self, bottom):
        if len(bottom) == 0:
            raise Exception("NonGrayMaskLayer should have inputs")
        # if an image has any (a,b) value which exceeds threshold, output 1
        return (np.sum(np.sum(np.sum(np.abs(bottom) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,na(),na(),na()]

class ClassRebalanceMultLayer(torch.nn.Module):
    ''' INPUTS
        bottom[0]   NxMxXxY     feature map
        bottom[1]   Nx1xXxY     boost coefficients
    OUTPUTS
        top[0]      NxMxXxY     on forward, gets copied from bottom[0]
    FUNCTIONALITY
        On forward pass, top[0] passes bottom[0]
        On backward pass, bottom[0] gets boosted by bottom[1]
        through pointwise multiplication (with singleton expansion) '''
    def __init__(self):
        super().__init__( )
    
    def forward(self, bottom, pp_factor):

        return bottom * pp_factor
        # top[0].data[...] = bottom[0].data[...]*bottom[1].data[...] # this was bad, would mess up the gradients going up


class Rebalance_Op(Function):
    @staticmethod
    def forward(ctx, input, factors):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input, factors)
        
        # return tensor * constant
        #return 不能仅返回 input 否则 不会执行 backward 操作,
        return input * 1.

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input, factors = ctx.saved_variables
        grad_input = grad_factors = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight.t())
            grad_input = grad_output * factors
            # grad_input = grad_output
        # if ctx.needs_input_grad[1]:
        #     #t() 转置, mm() matrix multiplication
        #     grad_factors = grad_output.t().mm(input)
        return grad_input, None


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor 
        获取feature map 通道上最大值对应的 prior probaility。
        例如，输入是 [1,313,255,255]
        输出是每个pixel的 313 个通道上的最大值在 prior_probs(313维) 的对应位置上的概率值。
        输出是 [1,1,255,255]
    '''
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        #prior_probs 不为 0 的元素在 uni_probs 相同位置全都赋值为 1
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print ('Prior factor correction:')
        print ('  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma))
        print ('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs)))

    def forward(self,data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out
