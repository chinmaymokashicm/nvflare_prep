# Blueprint to building production-level NVFlare architecture demo

## Features
- Production-level
- Pytorch-based image segmentation using [U-Net] 
    - 2D - [Source 1](https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3) and [Source 2](https://www.youtube.com/watch?v=u1loyDCoGbE&list=TLPQMDUxMTIwMjNdCX6-vkYGvA&index=1)
    - 3D - [Source 1](https://github.com/AghdamAmir/3D-UNet)


## Steps
1. Build a centralized model
2. Create an NVFlare simulation
3. Run on production mode

## Centralized Model
- [x] Write the NN architecture
- [ ] Load [dataset](https://blogs.kingston.ac.uk/retinal/chasedb1/)
- [ ] Train model
- [ ] Evaulate model
- [ ] Save weights

## NVFlare simulation
- [ ] Create folder structure
- [ ] Create config files
- [ ] Run simulation and save results

## Production
- [ ] Use provision tool
- [ ] Run startup kit
    - (Dockerized)
    - 1 overseer
    - 2 servers
    - 3 clients
- [ ] Run training and evaluation
- [ ] Save and evaluate