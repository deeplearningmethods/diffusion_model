# Diffusion Model
This repository contains code and resources related to the paper titled "..." by ... .
It is a simple implementation of diffusion model in  Pytorch. Depending on the input parameters <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model (DDPM)</a>, <a href="https://arxiv.org/abs/2102.09672">Improved DDPM</a>, <a href="https://arxiv.org/abs/2010.02502">Denoising Diffusion Implicit Model (DDIM)</a>, and <a href="https://arxiv.org/abs/2207.12598">classifier-free diffusion guidance</a> may be run.

<img src="./images/forward_diffusion.png" width="1000px"><img>

## Repository Structure

- `code/`: It contains the implementation of the diffusion models discussed in the paper. Use the file 'main.py' to train the Unet and generate new samples.
- `results/`: Results obtained from experiments conducted in the paper.
- `images/`: It contains the images empoyed in the paper and the code needed to generate them.

## How to Use

Run the main.py to train the UNet and visualize the results. The use of a GPU is strongly recommended.

## Next steps
The next steps are the implementations of:
- Inception score (IS)
- multiple GPU.


## Acknowlegements
This scheme was first proposed in [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

<!--## License

This project is licensed under the ... . -->
