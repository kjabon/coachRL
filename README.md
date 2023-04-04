# coachRL


## Installation/setup

For best results, follow these setup steps in order. This repo will only work on unix systems due to using Acme. This guide assumes Ubuntu. 
First, clone this repo to a local repository, then continue.

### CUDA installation
Download cuda for your distribution by making the appropriate selections at the link below. 
This may also yield commands you must run to update your keyring; for example my commands are:
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
```

However, stop short of actually running the ```install``` command.
https://developer.nvidia.com/cuda-downloads


Then you can do a clean update of cuda for your hardware:
```sh
sudo apt clean
sudo apt update
sudo apt purge cuda
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install cuda
sudo reboot #Yes, actually necessary
```

### CUDNN installation
Follow this guide. it's a bit time consuming, but it's all there.
You need to download it manually, dpkg, then apt install 3 things (cudnn, -dev, -samples)
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify




### Acme installation:
The pip version is not up to date, you must clone the github repo!
Follow step 4 from installation section of: https://github.com/deepmind/acme

### Other libraries
Finally, run setup.sh to install the rest of the necessary libraries.

