# Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration

This is the official implementation of Sec. B.3 in [Partial Distribution Matching via Partial Wasserstein Adversarial Networks](https://arxiv.org/abs/2409.10499),
where PWAN is used as a drop-in replacement of WGAN for robust image generation.

To make a WGAN model more robust to outliers (make it a PWAN model), 
you only need to
1. Use weight > 1 for the data sample. 
2. Make the output of discriminator negative, e.g., `output=-torch.abs(output)`

Training on Cifar10 dataset with a few mnist outliers:

| Training data                                    | WGAN                                         | PWAN                                                 | 
|--------------------------------------------------|----------------------------------------------|------------------------------------------------------|
 <img src="images\Real.png" width="256"/> | <img src="images\WGAN.png" width="256"/> | <img src="images\PWAN.png" width="256"/> 


## Usage
Plase see `a.txt` for usage.



## Reference


    @misc{wang2024partialdistributionmatchingpartial,
          title={Partial Distribution Matching via Partial Wasserstein Adversarial Networks}, 
          author={Zi-Ming Wang and Nan Xue and Ling Lei and Rebecka Jörnsten and Gui-Song Xia},
          year={2024},
          eprint={2409.10499},
          url={https://arxiv.org/abs/2409.10499}, 
    }

    @inproceedings{wang2022partial,
        title={Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration},
        author={Zi-Ming Wang and Nan Xue and Ling Lei and Gui-Song Xia},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2022}
    }

For any question, please contact me (wzm2256@gmail.com).


## LICENSE
The code is available under a [MIT license](LICENSE).
