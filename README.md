# coachRL


#For best results, follow these setup steps in order

###CUDA installation###
#Update nvidia cuda keyring by visiting the following website, and selecting your distribution
#https://developer.nvidia.com/cuda-downloads

#This yields commands to update your keyring; for me these commands are:
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
#sudo dpkg -i cuda-keyring_1.0-1_all.deb

#Then you can update cuda for your hardware:
#sudo apt clean
#sudo apt update
#sudo apt purge cuda
#sudo apt purge nvidia-*
#sudo apt autoremove
#sudo apt install cuda
#sudo reboot #Yes, actually necessary

###CUDNN installation###
# Follow this guide. it's a bit time consuming, but it's all there.
#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify
#(You need to download it manually, dpkg, then apt install 3 things (cudnn, -dev, -samples)

# Perhaps obvious to some: it is recommended you do the following within a virtual environment
# pip install --upgrade pip

#Acme installation:
# Step 4 from installation section of: https://github.com/deepmind/acme
# Pip version is not up to date, you must clone the github repo

#JAX with CUDA support installation:
#Follow this guide: https://github.com/google/jax/#installation
#Or just run this:
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#Access google api; you also need to use your google account and set up api access and tokens
#pip install google-api-python-client

#Tool to check nvidia gpu usage
# pip install GPUtil

#Tool to access iCloud from Python
# pip install pyicloud

#Access telegram bot api; you also need to use your telegram account and set up api access and tokens
# pip install aiogram
