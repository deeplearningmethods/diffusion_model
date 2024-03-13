# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L387

import torch
import math
from torch import einsum, nn
from einops import rearrange
import matplotlib.pyplot as plt

class SinusoidalPositionEmbeddings(nn.Module):
    ''' Takes a tensor of shape (batch_size, 1) as input,
     and turns this into a tensor of shape (batch_size, dim), 
     with dim being the dimensionality of the time embeddings. '''

    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def plot_sin_pos_emb(self, time = 1000, dim = 64):
        '''Plot the sinusoidal positional embeddings'''
        
        half_dim = self.dim // 2
        time = torch.tensor([time])
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block_down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, context_emb_dim, block_groups, heads=4, dim_head=16):
        super().__init__()
        ''' Perform element-wise multiplication and summation with the 'context_emb' and 'time_emb' tensors
        (transformation of 't' and 'c'), respectively and then apply two convolution operations followed by 
        group normalization to the input tensor 'x'.Finally is applied linear attention and a maxpool, this 
        one half the last two dimension.
        The output dimension of 'x2' is (batch_size, out_ch, W//2, H//2). We return also the residuals, needed in Expansive Phase of the Unet'''
     
        self.conv1 = nn.Conv2d(in_ch, out_ch,  3, padding=1)           # preserves last two dimensions, but modifies number of channel
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)           # preserves all dimensions
        self.bnorm1 = nn.GroupNorm(block_groups,out_ch)
        self.bnorm2 = nn.GroupNorm(block_groups,out_ch)
        self.maxpool = nn.MaxPool2d(2)                                 # halves last two dimensions
        self.act  = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, in_ch)
        if context_emb_dim != None:
            self.context_mlp = nn.Linear(context_emb_dim, in_ch)
        else:
            self.context_mlp = None
        self.attention = LinearAttention(out_ch, heads=4, dim_head=16) #Attention(out_ch, heads=4, dim_head=16)


    def forward(self, x, t, c ):
        '''
        x : tensor (batch_size, in_ch, W, H), 
        t : time vector (batch_size, time emb dimension),
        c : context vector (batch_size, context emb dimension)
        '''
        
        # Time and context embedding
        time_emb = self.act(self.time_mlp(t)) 
        context_emb = self.act(self.context_mlp(c)) 
        # Extend last 2 dimensions to have proper shape
        time_emb = time_emb[(..., ) + (None, ) * 2]
        context_emb = context_emb[(..., ) + (None, ) * 2]
        # Multiply and sum context and time embedding. Another possibility could be add a new channel with these information
        x2 = context_emb * x + time_emb 
        # First and second Conv
        x2 = self.act(self.bnorm1(self.conv1(x2)))
        x2 = self.act(self.bnorm2(self.conv2(x2)))
        # Self attention
        residual = self.attention(x2) + x2
        # Downsampling, max pooling
        x2 = self.maxpool(residual)

        return x2, residual


class Block_up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, context_emb_dim, block_groups, output_padding=0, heads=4, dim_head=16):
        super().__init__()
        ''' Apply self attention, convolution transpose, and then perform element-wise multiplication 
        and summation with the 'context_emb' and 'time_emb' tensors (transformation of 't' and 'c'), respectively.
        Finally are applied two convolutional layers, after the concatenation with the residual.
        The output dimension of 'x2' is (batch_size, out_ch, 2*W, 2*H). '''
        
        self.convtrans = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0, output_padding) # doubles last two dimensions
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)                         # preserves last two dimensions
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)                        # preserves all dimensions
        self.bnorm1 = nn.GroupNorm(block_groups,out_ch)
        self.bnorm2 = nn.GroupNorm(block_groups,out_ch)
        self.act  = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if context_emb_dim != None:
            self.context_mlp = nn.Linear(context_emb_dim, out_ch)
        else:
            self.context_mlp = None
        self.attention = LinearAttention(in_ch, heads=4, dim_head=16)  #if in_ch==init_conv_dim else Attention(in_ch, heads=4, dim_head=16) 

    def forward(self, x, t, residual, c ):
        '''
        x : tensor (batch_size, in_ch, W, H), 
        t : time vector (batch_size, time emb dimension), 
        residual : residual tensor (batch_size, out_ch, W, H),
        c : context vector (batch_size, context emb dimension)
        '''
        
        # Self attention
        x2 = self.attention(x) + x
        # Upsampling, convolutional transpose
        x2 = self.convtrans(x2)
        # Time and context embedding
        time_emb = self.act(self.time_mlp(t)) 
        context_emb = self.act(self.context_mlp(c))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        context_emb = context_emb[(..., ) + (None, ) * 2] 
        # Multiply and sum context and time embedding. Another possibility could be add a new channel with these information
        x2 = context_emb * x2 + time_emb
        # Concatenation of residual and considered tensor.  
        x2 = torch.cat((x2, residual), 1)  # Shape: (batch_size, out_ch*2=in_ch, 2*W, 2*H)
        # First and second Conv
        x2 = self.act(self.bnorm1(self.conv1(x2)))
        x2 = self.act(self.bnorm2(self.conv2(x2)))

        return x2
    

class Block_middle(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, context_emb_dim, block_groups):
        super().__init__()
        ''' Perform element-wise multiplication and summation with the 'context_emb' and 'time_emb' 
        tensors (transformation of 't' and 'c'), respectively. Then are applied two convolutional layers.
        The output dimension of 'x2' is (batch_size, out_ch, W, H). '''
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)                         # preserves last two dimensions
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)                        # preserves all dimensions
        self.bnorm1 = nn.GroupNorm(block_groups,out_ch)
        self.bnorm2 = nn.GroupNorm(block_groups,out_ch)
        self.act  = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, in_ch)
        if context_emb_dim != None:
            self.context_mlp = nn.Linear(context_emb_dim, in_ch)
        else:
            self.context_mlp = None

    def forward(self, x, t, c ):
        '''
        x : tensor (batch_size, in_ch, W, H), 
        t : time vector (batch_size, time emb dimension), 
        c : context vector (batch_size, context emb dimension)
        '''
        
        # Upsampling, convolutional transpose
        #x2 = self.convtrans(x2)  NOOOOO
        # Time and context embedding
        time_emb = self.act(self.time_mlp(t)) 
        context_emb = self.act(self.context_mlp(c))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        context_emb = context_emb[(..., ) + (None, ) * 2] 
        # Multiply and sum context and time embedding. Another possibility could be add a new channel with these information
        x2 = context_emb * x + time_emb
        # First and second Conv
        x2 = self.act(self.bnorm1(self.conv1(x2)))
        x2 = self.act(self.bnorm2(self.conv2(x2)))
        
        return x2


class LinearAttention(nn.Module):
    def __init__(
                self,
                dim,             # input dimension
                heads = 4,       # number of head in the attention layer 
                dim_head = 32,   # dimension of each head
                #num_mem_kv = 4
                ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads


        #self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        #x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        #mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        #k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, 
                 dim,                # input dimension
                 heads = 4,          # number of head in the attention layer 
                 dim_head = 32,      # dimension of each head 
                 # num_mem_kv=4
                 ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        
        
        # could be interesting adding this memory parameters
        #self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c ", h=self.heads), qkv)
        
        #mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        #k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)



class Unet(nn.Module):
    def __init__(self, in_channels = 3,         # channel of input images
                 init_conv_dim = 128,           # initial channel dimension of Unet
                 time_dim = 128,                # time dimension after sin embedding
                 context_dim = 128,             # context dimension after linear transformation
                 dim_factor = (1, 2, 4),        # scale factors of each Unet level
                 img_size = 64,                 # size of input images
                 num_classes = 1,               # number of classes
                 resnet_block_groups = 8,       # size of block groups
                 learned_variance = False,      # needed to pass from DDPM to improved DDPM (if True remember that the output dimension doubles)
                 attn_dim_head = 32,            # attention dim head
                 attn_heads = 4,                # attention head (pay attention to the dimension 32*4)
                 ):  
        
        super(Unet, self).__init__()
        
        # initialization
        self.num_classes = num_classes
        self.in_channels = in_channels 
        self.init_conv_dim = init_conv_dim
        self.time_dim = time_dim
        self.context_dim = context_dim
        self.img_size = img_size
    
        
        # Embedding of time step
        self.SPE_time = nn.Sequential(
        SinusoidalPositionEmbeddings(self.time_dim),
        nn.Linear(time_dim, time_dim),
        nn.GELU(),
        #nn.Linear(time_dim, time_dim)
        )
        
        
        # Embedding of context vector
        if num_classes>1:
            assert context_dim is not None, 'If the number of classes is greater than 1, then we have to define a context dimension'
            self.context_embed = nn.Sequential(
                nn.Linear(num_classes, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim)
                )
        else:
            assert context_dim is None, 'If there are not classes, then we must not define a context dimension'
            self.context_embed = None
        
        
        levels = len(dim_factor) # levels in the unet
        
        # Initialize the down-sampling part of the U-Net with len(dim_factor) levels
        # Define the down blocks based on the levels parameter and the dim_factor
        self.down_blocks = nn.ModuleList()
        for level in range(levels):
            input_dim = init_conv_dim * (dim_factor[level - 1]) if level > 0 else self.in_channels 
            output_dim = init_conv_dim * dim_factor[level]
            self.down_blocks.append(Block_down(input_dim, output_dim, time_dim,context_dim, resnet_block_groups))

    
        # Define the middle block
        self.middle = Block_middle(dim_factor[-1] * init_conv_dim, 2*dim_factor[-1] * init_conv_dim, \
                                   time_dim, context_dim, resnet_block_groups)
        #self.middle = nn.Sequential(
        #    nn.Conv2d(dim_factor[-1] * init_conv_dim, 2*dim_factor[-1] * init_conv_dim, 3, 1,1),   # 3x3 kernel with stride 1 and padding 1
        #    nn.GroupNorm(resnet_block_groups, 2*dim_factor[-1] * init_conv_dim),   # Group normalization
        #    nn.ReLU(),   # ReLU activation function,
        #    nn.Conv2d(2*dim_factor[-1] * init_conv_dim, 2*dim_factor[-1] * init_conv_dim, 3, 1,1),   # 3x3 kernel with stride 1 and padding 1
        #    nn.GroupNorm(resnet_block_groups, 2*dim_factor[-1] * init_conv_dim),   # Group normalization
        #    nn.ReLU(),   # ReLU activation function
        # )  

    
        # Initialize the up-sampling part of the U-Net with len(dim_factor) levels
        # PROBLEM: in down_block maxpool can lose a dimension if input is even (e.g. m=maxpool2d(2)===>m[1,1,7,7]=[1,1,3,3],
        # we need to add output padding parameters)
        self.up_blocks = nn.ModuleList()
        for level in range(levels):
            input_dim = init_conv_dim * dim_factor[levels -  level] if level>0 else 2 * init_conv_dim * dim_factor[-1]
            output_dim = init_conv_dim * dim_factor[levels - 1 - level]
            output_padding = 0 if img_size%(2**(levels-level))==0 else 1
            self.up_blocks.append(Block_up(input_dim, output_dim, time_dim, context_dim, resnet_block_groups, output_padding))
            
            
        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(dim_factor[0] *init_conv_dim, self.in_channels, 1,1), 
        )

    def forward(self, x, t, c):
        """
        x : (batch, channel, H, W) : input image
        t : (batch, time_dim)      : time step
        c : (batch, context_dim)    : context label, optional
        """
        t = self.SPE_time(t)
        c = self.context_embed(c)
        h = []
        # pass the result through the down-sampling path
        for down_block in self.down_blocks:
            x, residual = down_block(x,t,c)
            h.append(residual)
        
        
        x = self.middle(x,t,c) # add time? yes it is better
        for up_block in self.up_blocks:
            x = up_block(x, t, h.pop(), c)
        
        x = self.out(x)
        
        return x





