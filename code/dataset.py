import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class Dataset():
    def __init__(
        self,
        dataset_name,                           
        image_size = False,                     # dimension of images, e.g. (128,128)
        augment_horizontal_flip = False,        # horizontal rotation of some images
        selected_class = None,                  # employ only specific classes
        grayscale = False                       # convert images to gray scale
        #convert_image_to = None
    ):
        ''' Class needed to select and create a dataset applying eventually transformations.
        There is the to possibility to show samples. '''
         
         
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.selected_class = selected_class
        

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation = transforms.InterpolationMode.NEAREST) if image_size else nn.Identity(),   # resize image 
            transforms.Grayscale() if grayscale else nn.Identity(),                                                                 # black and white
            transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),                                        # random flip image if augment_horizontal_flip is True
            transforms.CenterCrop(image_size) if image_size else nn.Identity(),                                                                                      # crop images in the center
            transforms.ToTensor(),                                                                                                  # transfor images in tensor
            transforms.Lambda(lambda t: (t * 2) - 1)                                                                                # Scale between [-1, 1] instead of [0,1], better for the algorithm
        ])
        
        assert self.dataset_name in {'FGVCAIRCRAFT','Flowers102','StanfordCars','MNIST', 'CIFAR10','FashionMNIST'}, 'dataset name should be in (FGVCAIRCRAFT,Flowers102,StanfordCars,CIFAR10,FashionMNIST)'
        if self.dataset_name == 'FGVCAIRCRAFT':
            self.train = torchvision.datasets.FGVCAircraft(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.FGVCAircraft(root=".", download=True, transform=self.transform, split='test')
        if self.dataset_name == 'MNIST':
            self.train = torchvision.datasets.MNIST(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.MNIST(root=".", download=True, transform=self.transform, train=False)
        if self.dataset_name == 'FashionMNIST':
            self.train = torchvision.datasets.FashionMNIST(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.FashionMNIST(root=".", download=True, transform=self.transform, train=False)
        if self.dataset_name == 'Flowers102':
            self.train = torchvision.datasets.Flowers102(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.Flowers102(root=".", download=True, transform=self.transform, split='test')
        if self.dataset_name == 'StanfordCars':
            self.train = torchvision.datasets.StanfordCars(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.StanfordCars(root=".", download=True, transform=self.transform, split='test')
        if self.dataset_name == 'CIFAR10':
            self.train = torchvision.datasets.CIFAR10(root=".", download=True, transform=self.transform)
            self.test = torchvision.datasets.CIFAR10(root=".", download=True, transform=self.transform, train = 'False')
        
        # Select images of specific classes
        if selected_class != None:
            
            class_indices_train = [i for i, data in enumerate(self.train) if data[1] in selected_class]
            self.train = torch.utils.data.Subset(self.train, class_indices_train)
            class_indices_test = [i for i, data in enumerate(self.test) if data[1] in selected_class]
            self.test = torch.utils.data.Subset(self.test, class_indices_test)
            
    
    
    def show_images(self, num_samples=36, cols=6):
        """ Plots some samples from the dataset """
        plt.figure(figsize=(15,12))
        
        # reverse trnsformations
        reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                                                ])

        for i, img in enumerate(self.train):
            if i == num_samples:
                break
            plt.subplot(int(num_samples/cols) , cols, i + 1)  # create grid
            # correct input dimension. img is a tuple, in position 0 there is the image, in position 1 the label.
            if len(img[0].size())==4:
                print('there is dim 4')
                img[0] = img[0][None,:,:,:]
            if len(img[0].size())==3:
                # show image in gray scale or rgb scale
                if img[0].size()[0]==1:
                  plt.imshow(reverse_transforms(img[0]).squeeze(), cmap="gist_gray")
                else:
                    plt.imshow(reverse_transforms(img[0]))
            else:
                raise ValueError(f'Dimension of the image is not correct: {img.size()}') 
            plt.axis('off')  # Turn off axis
            
            
    def get_data(self,):
        """ Return the union of train and test dataset, in diffusion model the test dataset is not needed."""
        return torch.utils.data.ConcatDataset([self.train, self.test])
    
   