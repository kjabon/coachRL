
# Perhaps obvious to some: it is recommended you do the following within a virtual environment
# pip install --upgrade pip

#JAX with CUDA support installation:
#Follow this guide: https://github.com/google/jax/#installation
#You may run into some CUDA version issues
#TLDR; just run this:
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#Access google api; you also need to use your google account and set up api access and tokens
pip install google-api-python-client

#Tool to check nvidia gpu usage
pip install GPUtil

#Tool to access iCloud from Python
pip install pyicloud

#Access telegram bot api; you also need to use your telegram account and set up api access and tokens
pip install aiogram
