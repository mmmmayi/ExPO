'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''
import matplotlib.pyplot as plt
import math, torch, torchaudio, librosa.display
import torch.nn as nn
import torch.nn.functional as F
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )
        self.flipped_filter = torch.load('flipped_filter.pt')

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
        """
    def __init__(self,
                 in_features=256,
                 out_features=5994,
                 scale=32.0,
                 margin=0,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        #return cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        if mode =='score':
            return output
        else:
            return cosine*self.scale

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={},
                  margin={}, easy_margin={}'''.format(self.in_features,
                                                      self.out_features,
                                                      self.scale, self.margin,
                                                      self.easy_margin)

class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    def __init__(self, **kwargs):
        super(TSTP, self).__init__()

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats


class Speaker_resnet(nn.Module):
    def __init__(self,
                 num_blocks=[3,4,6,3],
                 m_channels=32,
                 feat_dim=80,
                 embed_dim=256,
                 pooling_func='TSTP'):
        super(Speaker_resnet, self).__init__()
        block = BasicBlock
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = False
        self.pre =  PreEmphasis()
        self.Spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2.0)
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1)
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = TSTP()
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, mode='feature'):
        with torch.no_grad():
            x = self.pre(x)
            x = self.Spec(x)
            frame_len = x.shape[-1]
            x = (self.Mel_scale(x)+1e-6).log()
            if frame_len%8>0:
                pad_num = math.ceil(frame_len/8)*8-frame_len
                pad = torch.nn.ZeroPad2d((0,pad_num,0,0))
                x = pad(x)

            #x = (self.Spec(x)+1e-8)
            #x = (self.Mel_scale(x)+1e-8).log()
            x = x[:,:,:200]
        #print(x.shape) #[128,80,1002]
        feature = x
        x = feature - torch.mean(feature, dim=-1, keepdim=True)
        #x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        #print('out.shape:',out.shape) #[B,32,80,T]
        out1 = self.layer1(out)
        #print('layer1 shape:',out.shape) #[B,32,80,T]
        out2 = self.layer2(out1)
        #print('layer2 shape:',out.shape) #[B,64,40,T/2]
        out3 = self.layer3(out2)
        #print('layer3 shape:',out.shape)#[B,128,20,T/4]
        out4 = self.layer4(out3)
        #print('layer4 shape:',out.shape)#[B,256,10,T/8]
        frame = out4
        stats = self.pool(out4)
        embed_a = self.seg_1(stats)
        '''
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return x, embed_a
        '''
        if mode=='feature':
            return frame
        elif mode == 'encoder':
            return [out1,out2,out3,out4, embed_a], feature
        elif mode == 'reference':
            return embed_a
        elif mode in ['score','loss']:
            score = self.projection(embed_a, targets, mode)
            result = torch.gather(score,1,targets.unsqueeze(1).long()).squeeze()
            if mode =='score':
                return result
            else:
                return score
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion*planes)
            )



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
          
        return out   
class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def CNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):

        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):

            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=int(kernel_size/2), bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)

def SubpixelUpsampler(in_channels, out_channels, num_blocks, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        transConv(in_channels, out_channels, num_blocks),
        PrintLayer()
        #activation_fn(),
    ]
    return nn.Sequential(*_modules)

def transConv(in_channels, out_channels, num_blocks, stride=2, follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):
    _modules = []
    strides = [stride] + [1] * (num_blocks - 1)
    out_padding=1
    for stride in strides:
        _modules.append(nn.ConvTranspose2d(in_channels, out_channels,3,stride=stride,padding=1,output_padding=out_padding))
        _modules.append(PrintLayer())
        _modules.append(nn.InstanceNorm2d(out_channels, affine=affine))
        _modules.append(PrintLayer())
        _modules.append(activation_fn())
        _modules.append(PrintLayer())
        in_channels = out_channels
        out_padding = 0
     
    return nn.Sequential(*_modules)

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        #print(torch.any(torch.isnan(x)))
        return x

class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,passthrough_channels, num_blocks, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels,out_channels=out_channels, num_blocks = num_blocks)
        self.follow_up = Block(out_channels+passthrough_channels,out_channels)
        self.norm = nn.InstanceNorm2d(out_channels)
    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out,self.norm(passthrough)), 1)
        return self.follow_up(out)




class BLSTM_enhance(nn.Module):
    
    def __init__(self):
        super(BLSTM_enhance, self).__init__()
        self.decoder = decoder()
        #self.speaker = Speaker_resnet()

        #path = "exp/resnet.pt"
        #checkpoint = torch.load(path)
        #self.speaker.load_state_dict(checkpoint, strict=False)

        #for p in self.speaker.parameters():
            #p.requires_grad = False
        #self.speaker.eval()        
    def forward(self,encoder_out):
        #self.speaker.eval()
        

        #encoder_out,feature = self.speaker(input,mode='encoder')
        mask = self.decoder(encoder_out)
 
        return mask

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        num_blocks = [3,6,4,3]
        self.uplayer4 = UpSampleBlock(in_channels=256,out_channels=128,passthrough_channels=128, num_blocks=num_blocks[-1])
        self.uplayer3 = UpSampleBlock(in_channels=128,out_channels=64,passthrough_channels=64, num_blocks=num_blocks[-2])
        self.uplayer2 = UpSampleBlock(in_channels=64,out_channels=32,passthrough_channels=32, num_blocks=num_blocks[-3])
        self.saliency_chans = nn.Conv2d(32,1,kernel_size=1,bias=False)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def binary(self,mask):
        return torch.where(mask>0,1,0)
    def forward(self,encoder_out):
        em = encoder_out[-1]
        scale4 = encoder_out[-2]
        #act = torch.sum(scale4*em.view(-1, 256, 1, 1), 1, keepdim=True)
        #th = torch.sigmoid(act)
        #scale4 = scale4*th

        upsample3 = self.uplayer4(scale4, encoder_out[-3])
        upsample2 = self.uplayer3(upsample3, encoder_out[-4])
        upsample1 = self.uplayer2(upsample2, encoder_out[-5])
        saliency_chans = self.saliency_chans(upsample1)
        #a = torch.abs(saliency_chans[:,0,:,:])
        #b = torch.abs(saliency_chans[:,1,:,:])
        return self.sig(saliency_chans)


