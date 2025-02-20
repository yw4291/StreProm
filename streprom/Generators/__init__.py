from .wgan import WGAN
from .gan_language_forGA import WGAN_GP
from .vae import VAE
from .aae import AAE
# from .aae_supervise_latent_onehot import AAE_supervise_onehot
# from .aae_supervise_latent_bin import AAE_supervise_bin
from .aae_semi import AAE_semi
from .aae_semi_d2crossentropy import AAE_semi_D_crossentropy
#from .aae_semi_convmaxpool_noresblock import AAE_semi_convmaxpool_noresblock
from .aae_semi_maxpool_resblock import AAE_semi_maxpool_resblock
from .aae_semi_maxpooladd import AAE_semi_maxpooladd
from .aae_Noy_test import AAE_Noy_test