import os

promoter_len=80
log_dir = './log'
generative_model_dir=os.path.join(log_dir,'generative_model')
sample_dir=os.path.join(log_dir,'designed_sequences')
evalu_dir=os.path.join(log_dir,'evalu/')
rep_dir=os.path.join(log_dir,'latent_representation/')
GA_dir=os.path.join(log_dir,'GA/')
pred_weight_dir_evalu_GA=os.path.join(log_dir,'predictor_weight_dir_evalu/')
pred_weight_dir_evalu_GD=os.path.join(log_dir,'predictor_weight_dir_evalu/')
predictor_trainop_dir=os.path.join(log_dir,'predictor_trainop/')
predictor_evaluop_dir=os.path.join(log_dir,'predictor_evaluop/')




