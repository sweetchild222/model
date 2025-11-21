from torch import nn
from .attention_pool import AttentionPool
from .bottleneck import Bottleneck


class ImageEncoder(nn.Module):


    def __init__(self, resolution, width, output_dim):
        super().__init__()
        
        self.stem_sequential = self.create_stem_sequential(width)

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        
        self.bottleneck_sequential = self.create_bottleneck_sequential(width)

        spacial_dim = resolution // 32
        embed_dim = width * 32
        num_heads = embed_dim // 64

        self.attention_pool = AttentionPool(spacial_dim, embed_dim=embed_dim, num_heads=num_heads, output_dim=output_dim)


    def create_stem_sequential(self, width):
        
        outplanes = width//2

        params= [{'inplanes':3, 'outplanes':outplanes, 'kernel_size':3, 'stride':2, 'padding':1},
                 {'inplanes':outplanes, 'outplanes':outplanes, 'kernel_size':3, 'stride':1, 'padding':1},
                 {'inplanes':outplanes, 'outplanes':width, 'kernel_size':3, 'stride':1, 'padding':1}]
        
        sequential = nn.Sequential()

        for param in params:

            block = self.crete_stem_block(**param)

            sequential.append(block)
            
        return sequential
    

    def crete_stem_block(self, inplanes, outplanes, kernel_size, stride, padding):

        conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        batch_norm = nn.BatchNorm2d(outplanes)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(*[conv, batch_norm, relu])


    def create_bottleneck_sequential(self, width):

        expansion = 4

        params = [{'inplanes':width, 'midplanes': width*(2**0), 'outplanes':width*(2**0)*expansion, 'layer_count': 2, 'stride': 1},
                  {'inplanes':width*(2**0)*expansion, 'midplanes': width*(2**1), 'outplanes':width*(2**1)*expansion, 'layer_count': 3, 'stride': 2},
                  {'inplanes':width*(2**1)*expansion, 'midplanes': width*(2**2), 'outplanes':width*(2**2)*expansion, 'layer_count': 5, 'stride': 2},
                  {'inplanes':width*(2**2)*expansion, 'midplanes': width*(2**3), 'outplanes':width*(2**3)*expansion, 'layer_count': 2, 'stride': 2}]

        sequential = nn.Sequential()

        for param in params:
            block = self.create_bottleneck_block(**param)
            sequential.append(block)

        return sequential


    def create_bottleneck_block(self, inplanes, midplanes, outplanes, layer_count, stride):

        front_bottleneck = Bottleneck(inplanes, midplanes, outplanes, stride)
        
        bottlenecks = [Bottleneck(outplanes, midplanes, outplanes, 1) for _ in range(layer_count-1)]
        
        return nn.Sequential(*([front_bottleneck] + bottlenecks))
    

    def forward(self, x):
        
        x = self.stem_sequential(x)

        x = self.avg_pool(x)

        x = self.bottleneck_sequential(x)
        
        x = self.attention_pool(x)

        return x
    


