from skimage import color

import torch
import models.pytorch.model_pool as model
import numpy as np
import matplotlib.pyplot as plt

gpu_id = -1
path = ''
dist = False
print('path = %s' % path)
print('Model set! dist mode? ', dist)
net = model.SIGGRAPHGenerator(dist=dist)
state_dict = torch.load('models/pytorch/model.pth')
if hasattr(state_dict, '_metadata'):
    del state_dict._metadata

# net=net.to('cuda')
# patch InstanceNorm checkpoints prior to 0.4

dtype = torch.FloatTensor

img_l_mc = np.load('test_imgs/huidutu/self.img_l_mc.npy')
img_ab_mc = np.load('test_imgs/huidutu/self.input_ab_mc.npy')
img_mask_mult = np.load('test_imgs/huidutu/self.input_mask_mult.npy')
output_ab1 = np.load('test_imgs/huidutu/output_ab.npy')

# net.eval()


img_l_mc = torch.from_numpy(img_l_mc)
img_l_mc = img_l_mc.type(torch.FloatTensor)
# img_l_mc = img_l_mc.cuda()

img_ab_mc = torch.from_numpy(img_ab_mc)
img_ab_mc = img_ab_mc.type(torch.FloatTensor)
# img_ab_mc = img_ab_mc.cuda()

img_mask_mult = torch.from_numpy(img_mask_mult)
img_mask_mult = img_mask_mult.type(torch.FloatTensor)
# img_mask_mult = img_mask_mult.cuda()

output_ab = net.forward(img_l_mc, img_ab_mc, img_mask_mult)[0, :, :, :].cpu().data.numpy()


def lab2rgb_transpose(img_l, img_ab):
    ''' INPUTS
            img_l     1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    # pre=color.lab2rgb(pred_lab)
    # pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
    return pred_lab


print(output_ab)
output_rgb = lab2rgb_transpose(img_l_mc, output_ab1)
plt.plot(output_ab[0,:,:],'gray')
plt.show()
plt.plot(output_ab1[0,:,:],'gray')
plt.show()
plt.plot(output_rgb)
plt.show()
