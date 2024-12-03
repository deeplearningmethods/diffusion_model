"""
Created on Fri Oct 13 16:33:35 2023

@author: Davide
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy import stats


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.002):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / timesteps
        return torch.linspace(beta_start * scale, beta_end * scale, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in arXiv:2102.09672
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
def get_index_from_list(vals, t, x_shape, device):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.cpu().gather(-1, t.cpu()) # extract from last dimension the index t
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


@torch.inference_mode()
def reverse_transforms_image(image):
        """ Inverse of the initial transformation """
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            #transforms.ToPILImage(),
        ])
    
        image = reverse_transforms(image.detach().cpu())
        return image
    
@torch.inference_mode()
def show_tensor_image(image): 
        """ Plot tensor image """
        image = reverse_transforms_image(image)
        if len(image.shape) == 4:
            raise ValueError('It has been passed a batch of image instead of a single image')
        image = np.squeeze(image) # eliminate useless dimensions
        if len(image.shape)==2:   # if image has only one channel now the len of the shape is 2
            plt.imshow(image, cmap='gist_gray')
        else:
            plt.imshow(image)
    

# implement different objective function, min_snr_loss_weight. Improve movement to device
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,                       # unet
        device,                      # device where the computation are done
        *,
        timesteps = 1000,            # training timestep
        sampling_timesteps = None,   # sampling timestep
        objective = 'pred_noise',
        beta_schedule = 'linear',
        schedule_fn_kwargs = dict(),
        loss_function = 'mse',
        ddim_sampling_eta = 1.,
        w = 0,                       # constant needed for classifier free guidance
        p_uncond = 0,
        offset_noise_strength = 0.1,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # signal noise ratio https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
        ):
        """ 
        model : unet model
        device : device where the computation are done. cuda or gpu
        timestep : training timestep, at least 1000.
        sampling_timesteps : sampling timestep. By default like timestep
        objective : prediction of the model. 'pred_noise', 'pred_x0', 'pred_v'
        beta_schedule : scheduler needed to define betas. 'linear', 'cosine'
        schedule_fn_kwargs :
        loss_function : the loss function. 'mse', 'l1'
        ddim_sampling_eta : if 1 then we have a DDPM, if 0 we have DDIM. Any number between 0 and 1 means an interpolation between DDPM and DDIM
        w : when w>0, we utilize classifier-free guidance, when w=0 model is DDPM. As w increases, we are removing more “null” images. w should be in [0,10]
        p_uncond : use it only if we are using classifier free guidance! Probability of training a null class for classifier free guidance. In [0,1), usually 0.1 or 0.2
        """
        super().__init__()
        

        self.model = model
        self.device = device
        self.in_channels = self.model.in_channels
        self.image_size = self.model.img_size
        
        assert loss_function in {'mse', 'l1'}, 'loss function should be mse or l1'
        if loss_function == 'mse':
            self.loss_function = nn.MSELoss(reduction='none')
        elif loss_function == 'l1':
            self.loss_function = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f'unknown loss_function {loss_function}')
        
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # cumulative product of previous elements
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.) # add value "1" at first position and shift others values of the position (last value is removed)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # calculations for posterior q(x_{t-1} | x_t, x_0)
                

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        # sampling related parameters
        if sampling_timesteps is not None:
            self.sampling_timesteps =  sampling_timesteps 
        else:
            self.sampling_timesteps =  timesteps # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps, 'Sampling timesteps must be lower or equal tan training timesteps'
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        # if eta is big than 0 then 
        if ddim_sampling_eta==1:
            assert not self.is_ddim_sampling,  'Using DDPM eta we should not skip timesteps in the sampling phase'

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))


        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength, 0.1 was ideal
        self.offset_noise_strength = offset_noise_strength
        
        # derive loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)


        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))


        # initialize classifier free guidance parameters
        self.w = w  
        self.p_uncond = p_uncond
        assert (self.w==0 and self.p_uncond==0) or self.w>0 and (self.p_uncond>0), 'Classifier free guidance parameters are not specify correctly'




    def forward_diffusion_sample(self,x_0, t, noise):
        """
        Takes an image, a timestep and some noise as input and
        returns the noisy version of it
        """
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape, self.device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, self.device
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
        + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)


    def p_losses(self, x_0, t, context = None, noise = None):
        """
        Takes an image and a timestep as input, calculate the noise versione version of it,
        uses the model to predict the original original image from the noisy one,
        and return the loss function between the output of the model and the target.
        """
        b, c, h, w = x_0.shape
        # random noise
        noise = torch.randn_like(x_0, device = self.device)
        
        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(b, c, 1, 1, device = self.device)
            noise += self.offset_noise_strength * offset_noise

        # noisy image 
        x = self.forward_diffusion_sample(x_0 = x_0, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        #x_self_cond = None
        #if self.self_condition and np.random.random() < 0.5:
        #    with torch.inference_mode():
        #        x_self_cond = self.model_predictions(x, t).pred_x_start
        #        x_self_cond.detach_()

        # predict model outout
        model_out = self.model(x, t, context)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_0
        elif self.objective == 'pred_v':
            v = self.predict_v(x_0, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        # loss
        loss = self.loss_function(model_out, target)
        
        return loss.mean()


    def forward(self, img, t, *args, **kwargs):
        b, c, h, w, img_size, = *img.shape, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        
        return self.p_losses(img, t, *args, **kwargs)
    
    
    @torch.inference_mode()
    def simulate_forward_diffusion(self, real_image, num_images = 10, hist = False):  
        """
        Simulate a forward diffusion. Given an image we add progressevely some noise plotting the results
        """
        # Simulate forward diffusion
        plt.axis('off')
        T = self.num_timesteps
        stepsize = int(T/num_images)
        if self.model.in_channels == 1: # needed for black and white images
            style = 'gist_gray' 
        else:
            style = None
        fig, axs = plt.subplots(1, num_images+1, figsize=(30,4))
        plt.subplots_adjust(wspace=0)  # Set spacing between subplots to zero
        plt.axis('off')
        fig.suptitle('Forward process')
        if hist:
            fig_hist, axs_hist = plt.subplots(1, num_images+1, figsize=(30,4))
            axs_hist[0].set_title('Starting image')
            x = np.linspace(-3, 3, 100)
            density = stats.gaussian_kde(real_image.flatten(), bw_method='scott')
            axs_hist[0].plot(x, density(x), color='blue')  
            axs_hist[0].set_yticklabels([])
            axs_hist[0].set_xticklabels([])
            #axs_hist[0].hist(real_image.squeeze().flatten())
            #axs_hist[0].set_ylim(0, 0.035)
            
        axs[0].imshow(reverse_transforms_image(real_image).squeeze(), cmap = style )
        axs[0].set_title('Starting image')
        axs[0].axis('off')
        
        for idx in range(stepsize, T+stepsize, stepsize):
            t = torch.Tensor([idx-1]).type(torch.int64) # in the code step 1 happens when t=0
            img_index = idx//stepsize
            img = self.forward_diffusion_sample(real_image, t, torch.randn_like(real_image)).cpu()
            axs[int(img_index)].imshow(reverse_transforms_image(img), cmap = style )
            axs[int(img_index)].set_title(f't={idx}')
            axs[int(img_index)].axis('off')
            
            if hist:
                density = stats.gaussian_kde(img.flatten(), bw_method='scott')
                axs_hist[int(img_index)].plot(x, density(x), color='blue')  
                #axs_hist[int(img_index)].hist(img.flatten(),)# density=True )
                #axs_hist[int(img_index)].set_ylim(0, 0.035)
                axs_hist[int(img_index)].set_title(f't={idx}')
                axs_hist[int(img_index)].set_yticklabels([])
                axs_hist[int(img_index)].set_xticklabels([])
            plt.subplots_adjust( hspace=0.5)
        plt.show()
        
            
        
    @torch.inference_mode()
    def sample_timestep(self, x, t, c, prev_t = None):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image and the image at the previous time step.
        Applies noise to this image, if we are not in the last step yet.
        """
        # if self.objective is 'pred_noise'
        betas_t = get_index_from_list(self.betas, t, x.shape, self.device)
        alphas_cumprod_t_minus_one = get_index_from_list(self.alphas_cumprod, prev_t, x.shape, self.device) 
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape, self.device)
        sqrt_alphas_cumprod_t_minus_one = get_index_from_list(self.sqrt_alphas_cumprod, prev_t, x.shape, self.device) 
        eta = self.ddim_sampling_eta
        
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape, self.device)
        sqrt_one_minus_alphas_cumprod_t_minus_one = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, prev_t, x.shape, self.device) 
        
        sigma = eta * torch.sqrt(betas_t) * sqrt_one_minus_alphas_cumprod_t_minus_one / sqrt_one_minus_alphas_cumprod_t
        const = torch.sqrt(1- alphas_cumprod_t_minus_one-sigma**2)
        
        pred_noise = self.model(x, t, c)
        # if classifier free guidance is used, then modify the pred_noise
        if self.w > 0:
            pred_noise = (1 + self.w) * pred_noise - self.w * self.model(x, t, 0*c)
            
        x_start = 1/sqrt_alphas_cumprod_t *( x - sqrt_one_minus_alphas_cumprod_t * pred_noise)
        x_prec =  sqrt_alphas_cumprod_t_minus_one * x_start + const * pred_noise 
        
        
        # if self.objective is 'pred_noise'
        #betas_t = get_index_from_list(self.betas, t, x.shape, self.device)
        #sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        #    self.sqrt_one_minus_alphas_cumprod, t, x.shape, self.device
        #)
        #sqrt_recip_alphas_t = get_index_from_list(torch.sqrt(1.0 /(1-self.betas)), t, x.shape, self.device)
        # Call model (current image - noise prediction)
        #model_mean = sqrt_recip_alphas_t * (
        #    x - betas_t * self.model(x, t, c) / sqrt_one_minus_alphas_cumprod_t #x_self_cond=None
        #)
        #posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape, self.device)
    
        if t == 0:
            return x_start, x_prec, x_prec
        else:
            noise = torch.randn_like(x)
            return x_start, x_prec, x_prec + sigma * noise

    @torch.inference_mode()
    def sample_plot_image(self, epoch=-1, desired_class = None, custom_entry = None, num_images=10, save = False):
        """
        Calls the model to do the backward process from a noisy image. There is the possibility to select a random class, otherwise it is selected randomly.
        Returns 'num_images' images from the initial random image to the denoised one.
        """
        assert custom_entry is not None or ( desired_class in range(self.model.num_classes)) or desired_class == None, 'Class does not exist'
        self.model.eval()
        if custom_entry is not None:
            c = custom_entry
        else:
            # selects the desired class or take a random one
            if desired_class == None:
                desired_class = torch.randint(0,self.model.num_classes,(1,), device = self.device)
            else:
                desired_class = torch.tensor(desired_class, device = self.device)
            c = torch.nn.functional.one_hot(desired_class, num_classes=self.model.num_classes).float()
       
        
        # Sample noise
        T = self.num_timesteps
        img = torch.randn((1, self.model.in_channels, self.image_size, self.image_size), device=self.device)
        
        # sampling timestep, for DDIM we could have a number smaller than T
        times = torch.linspace(0,T-1, self.sampling_timesteps)
        times = times.int().tolist()

        
        # create figures
        stepsize = int(self.sampling_timesteps/num_images)
        if self.model.in_channels == 1: # needed for black and white images
            style = 'gist_gray' 
        else:
            style = None
        fig1, axs1 = plt.subplots(1, num_images+1, figsize=(30,4))
        fig2, axs2 = plt.subplots(1, num_images+1, figsize=(30,4))
        fig1.suptitle('Backward process')
        fig2.suptitle('Reconstruction of final image from time step t')
        axs1[num_images].imshow(reverse_transforms_image(img.squeeze(0)), cmap = style )
        axs1[num_images].set_title('Starting image')
        axs1[num_images].axis('off')
        axs2[num_images].imshow(reverse_transforms_image(img.squeeze(0)), cmap = style)
        axs2[num_images].set_title('Starting image')
        axs2[num_images].axis('off')
        
        for i in range(0,self.sampling_timesteps)[::-1]:
            t = torch.full((1,), times[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((1,), times[i-1] if i>0 else 0, device=self.device, dtype=torch.long)
            img_start, img_without_noise, img = self.sample_timestep(img, t, c, prev_t)
            # This is to maintain the natural range of the distribution
            # Calculate dynamic thresholds using percentiles (like 99.5th and 0.5th)
            max_val = torch.quantile(img, 0.995)  # Upper dynamic threshold (99.5th percentile)
            min_val = torch.quantile(img, 0.005)  # Lower dynamic threshold (0.5th percentile)
            # Clip predicted image values within the dynamic range
            img = torch.clamp(img, min_val, max_val)
            #img = torch.clamp(img, -1.0, 1.0)
            img_start = torch.clamp(img_start, -1.0, 1.0)
            # plot the image every 'stepsize' steps
            if i % stepsize == 0:
                axs1[int(times[i]/stepsize)].imshow(reverse_transforms_image(img.squeeze(0)), cmap = style)
                axs1[int(times[i]/stepsize)].set_title(f't={i}')
                axs1[int(times[i]/stepsize)].axis('off')
                axs2[int(times[i]/stepsize)].imshow(reverse_transforms_image(img_start.squeeze(0)), cmap = style)
                axs2[int(times[i]/stepsize)].set_title(f't={i}')
                axs2[int(times[i]/stepsize)].axis('off')

            
        fig1.subplots_adjust(hspace=0.5, wspace=0)
        fig2.subplots_adjust(hspace=0.5, wspace=0)
        if save == True:
            plt.savefig('images/img_{}'.format(epoch))
        plt.show()

        return img

    @torch.inference_mode()
    def sample_plot_image_grid(self, desired_class=[None], custom_entry=None, grid_size=3, save = False):
        """
        Generates a 3x3 grid of unique denoised images by calling the 'sample_plot_image' function 9 times.
        """
        if desired_class == None:
            desired_class = [None] * (grid_size**2)
        if custom_entry == None:
            custom_entry = [None] * (grid_size**2)
        
        # Sample noise
        T = self.num_timesteps
        
        # sampling timestep, for DDIM we could have a number smaller than T
        times = torch.linspace(0,T-1, self.sampling_timesteps)
        times = times.int().tolist()
        
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle('Generated Images')
    
        # Loop to generate each image in the grid
        for row in range(grid_size):
            for col in range(grid_size):               
                img = torch.randn((1, self.model.in_channels, self.image_size, self.image_size), device=self.device)
                self.model.eval()
                if custom_entry[row*grid_size + col] is not None:
                    c = custom_entry[row*grid_size + col]
                else:
                    # selects the desired class or take a random one
                    if desired_class[row*grid_size + col] == None:
                        desired_class[row*grid_size + col] = torch.randint(0,self.model.num_classes,(1,), device = self.device)
                    else:
                        assert desired_class[row*grid_size + col] in range(self.model.num_classes), 'Class does not exist'
                        desired_class[row*grid_size + col] = torch.tensor(desired_class[row*grid_size + col], device = self.device)
                    c = torch.nn.functional.one_hot(desired_class[row*grid_size + col], num_classes=self.model.num_classes).float()
                
                # Generate a unique image
                for i in range(0,self.sampling_timesteps)[::-1]:
                    t = torch.full((1,), times[i], device=self.device, dtype=torch.long)
                    prev_t = torch.full((1,), times[i-1] if i>0 else 0, device=self.device, dtype=torch.long)
                    img_start, img_without_noise, img = self.sample_timestep(img, t, c, prev_t)
                    # This is to maintain the natural range of the distribution
                    # Calculate dynamic thresholds using percentiles (like 99.5th and 0.5th)
                    max_val = torch.quantile(img, 0.995)  # Upper dynamic threshold (99.5th percentile)
                    min_val = torch.quantile(img, 0.005)  # Lower dynamic threshold (0.5th percentile)
                    # Clip predicted image values within the dynamic range
                    img = torch.clamp(img, min_val, max_val)
                    img = torch.clamp(img, -1.0, 1.0)
                    img_start = torch.clamp(img_start, -1.0, 1.0)
                
                # Display the generated image in the grid
                axs[row, col].imshow(reverse_transforms_image(img.squeeze(0)), cmap='gist_gray' if self.model.in_channels == 1 else None)
                axs[row, col].axis('off')
        
        # Adjust layout and save/show image
        fig.subplots_adjust(hspace=0, wspace=0)
        if save == True:
            plt.savefig('images/img_generated')
        plt.show()
