import torch.nn as nn
from model import swinOIR

def make_model(args, parent=False):
    return model(args)

class model(nn.Module):

    def __init__(self, args):
        super(model, self).__init__()

        upscale = 4
        window_size = 8
        height = (1024 // upscale // window_size + 1) * window_size
        width = (720 // upscale // window_size + 1) * window_size


        self.transformer = swinOIR.swinOIR(upscale=4, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[4, 4, 4, 4],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
                   
    def forward(self, x):
        
        output = self.transformer(x)

        return output


