import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np
import tensorflow.compat.v1 as tf
from .language_helpers import *
import pickle
#import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
#from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D
#from tensorflow.keras.layers import *
from tensorflow.keras.layers import *
from ..ProcessData import oh2seq
import math
from .language_helpers import NgramLanguageModel,load_dataset,seq2oh
tf.disable_v2_behavior()



class WGAN_GP:
    def __init__(self, 
                 inputfile=None,
                 SEQ_LEN=80,
                 log_dir='./'
                 ):
        self.SEQ_LEN = SEQ_LEN
        self.log_dir = log_dir 
        self.MAX_N_EXAMPLES = 14098 
        # log_dir='./log_7_18_'+str(SEQ_LEN)+'bp' #3_24
        # if os.path.exists(log_dir) == False:
        #     os.makedirs(log_dir)
        self.inputfile = inputfile#'spe6_NAR_primary_TK24_200bp_fasta_'+str(SEQ_LEN)+'bp_nounk.txt'
        cur_path = os.getcwd()
        self.DATA_DIR = os.path.join(cur_path,'seq')
        lib.print_model_settings(locals().copy())
        self.lines, self.charmap, self.inv_charmap = load_dataset( #language_helpers.load_dataset
            filename=self.inputfile,
            max_length=self.SEQ_LEN,
            max_n_examples=self.MAX_N_EXAMPLES,
            data_dir=self.DATA_DIR
        )
        print('charmap:')
        print(self.charmap)

    def softmax(self,logits):
        return tf.reshape(
            tf.nn.softmax(
                tf.reshape(logits, [-1, len(self.charmap)])
            ),
            tf.shape(logits)
        )
    def make_noise(self,shape):
        return tf.random_normal(shape)

    def ResBlock(self,name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.1', self.DIM, self.DIM, 3, output) #6 5
        output = tf.nn.relu(output)
        output = lib.ops.conv1d.Conv1D(name+'.2', self.DIM, self.DIM, 3, output) #6 5
        return inputs + (0.3*output)
    #    return 0.3 * output

    def GeneratorNet(self,z, reuse=False):#n_samples
        #bs,128#output = make_noise(shape=[n_samples, 128])
        with tf.variable_scope("Generator", reuse=reuse):
            output = lib.ops.linear.Linear('Generator.Input', 128, self.SEQ_LEN*self.DIM, z)#bs,512*200
            output = tf.reshape(output, [-1, self.DIM, self.SEQ_LEN]) #bs,512,200
            output = self.ResBlock('Generator.1', output) #
            output = self.ResBlock('Generator.2', output)
            output = self.ResBlock('Generator.3', output)
            output = self.ResBlock('Generator.4', output)
            output = self.ResBlock('Generator.5', output) #bs,512,200
            output = lib.ops.conv1d.Conv1D('Generator.Output', self.DIM, len(self.charmap), 1, output) #bs,4,200
            output = tf.transpose(output, [0, 2, 1])
            output = self.softmax(output)
            return output


    def DiscriminatorNet(self,inputs,reuse=False):
        with tf.variable_scope("Discriminator", reuse=reuse):
            output = tf.transpose(inputs, [0,2,1])
            output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(self.charmap), self.DIM, 1, output)
            output = self.ResBlock('Discriminator.1', output)
            output = self.ResBlock('Discriminator.2', output)
            output = self.ResBlock('Discriminator.3', output)
            output = self.ResBlock('Discriminator.4', output)
            output = self.ResBlock('Discriminator.5', output)
            output = tf.reshape(output, [-1, self.SEQ_LEN*self.DIM])
            output = lib.ops.linear.Linear('Discriminator.Output', self.SEQ_LEN*self.DIM, 1, output)
            return output


    def Generator(self,Ntime=10,z=None,seed=1,n=1):
        if z is None:
            np.random.seed(seed)
            z = np.random.normal(size=(self.BATCH_SIZE*Ntime,self.Z_DIM))#gen_batchsize_time=10,
        generated_seq = []
        num = z.shape[0]
        #print(' [*] Generate num:'+str(num))
        batches = math.ceil(num/self.BATCH_SIZE)-1
        #T1 = time.time()
        for b in range(batches):
            oh = self.sess.run(self.fake_inputs,feed_dict={self.z:z[b*self.BATCH_SIZE:(b+1)*self.BATCH_SIZE,:]})
            generated_seq.extend(oh2seq(oh,self.invcharmap))
        #T2 = time.time()
        with open('./genseq_onebillion_divide_'+str(n), 'ab') as f:
                    pickle.dump(generated_seq, f) 
        return #generated_seq

    # Dataset iterator
    def inf_train_gen(self):
        while True:
            np.random.shuffle(self.lines)
            for i in range(0, len(self.lines)-self.BATCH_SIZE+1, self.BATCH_SIZE):
                yield np.array(
                    [[self.charmap[c] for c in l] for l in self.lines[i:i+self.BATCH_SIZE]], 
                    dtype='int32'
                )

    def generate_samples(self,fake_inputs):
                #2.
                #samples = session.run(fake_inputs)
                samples = self.sess.run(fake_inputs)
                samples = np.argmax(samples, axis=2)
                decoded_samples = []
                for i in range(len(samples)):
                    decoded = []
                    for j in range(len(samples[i])):
                        decoded.append(self.inv_charmap[samples[i][j]])
                    decoded_samples.append(tuple(decoded))
                return decoded_samples

    # During training we monitor JS divergence between the true & generated ngram
    # distributions for n=1,2,3,4. To get an idea of the optimal values, we
    # evaluate these statistics on a held-out set first.

    def gradient_penalty(self, real, fake):
        alpha = tf.random_uniform(
                shape=[self.BATCH_SIZE,1,1], 
                minval=0.,
                maxval=1.
                )
        differences = fake - real
        interpolates = real + (alpha*differences)
        gradients = tf.gradients(self.DiscriminatorNet(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty
    
    def BuildModel(self,
                   BATCH_SIZE=32,
                   DIM=512,
                   Z_DIM=128,
                   LAMBDA=10,
                   CRITIC_ITERS=5,
                   model_name='wgan_whc'):
        self.BATCH_SIZE = BATCH_SIZE # Batch size
        self.DIM = DIM 
        self.Z_DIM =Z_DIM
        self.LAMBDA = LAMBDA # Gradient penalty lambda hyperparameter.
        self.model_name = model_name
        self.CRITIC_ITERS = CRITIC_ITERS
        self.true_char_ngram_lms = [NgramLanguageModel(i+1, self.lines[10*self.BATCH_SIZE:], tokenize=False) for i in range(4)]#language_helpers.
        validation_char_ngram_lms = [NgramLanguageModel(i+1, self.lines[:10*self.BATCH_SIZE], tokenize=False) for i in range(4)]#language_helpers.
        for i in range(4):
            print("validation set JSD for n={}: {}".format(i+1, self.true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))#language_helpers.
        self.true_char_ngram_lms = [NgramLanguageModel(i+1, self.lines, tokenize=False) for i in range(4)]

        #
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess =tf.Session(config=config,graph=self.graph)


        #
        #with tf.Session() as session:
        with self.graph.as_default():
            print('Building model...')
            self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, self.SEQ_LEN])
            self.real_inputs = tf.one_hot(self.real_inputs_discrete, len(self.charmap))
            self.z = self.make_noise(shape=[self.BATCH_SIZE, self.Z_DIM])
            self.fake_inputs = self.GeneratorNet(self.z,reuse=True)#BATCH_SIZE
            self.fake_inputs_discrete = tf.argmax(self.fake_inputs, self.fake_inputs.get_shape().ndims-1)

            disc_real = self.DiscriminatorNet(self.real_inputs) 
            disc_fake = self.DiscriminatorNet(self.fake_inputs,reuse=True)

            self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
            self.gen_cost = -tf.reduce_mean(disc_fake)
            #
            # WGAN lipschitz-penalty
            # alpha = tf.random_uniform(
            #     shape=[self.BATCH_SIZE,1,1], 
            #     minval=0.,
            #     maxval=1.
            # )
            # differences = self.fake_inputs - real_inputs
            # interpolates = real_inputs + (alpha*differences)
            # gradients = tf.gradients(self.Discriminator(interpolates), [interpolates])[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
            # gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            GP = self.gradient_penalty(self.real_inputs, self.fake_inputs)
            self.disc_cost += self.LAMBDA*GP
            self.saver = tf.train.Saver(max_to_keep=50)
        return 
 
    def Train(self,ITERS= 200000,sample_dir='./',checkpoint_dir='./'):
        self.ITERS=ITERS
        self.sample_dir = sample_dir
        self.checkpoint_dir = checkpoint_dir
        with self.graph.as_default():
            gen_params = lib.params_with_name('Generator')
            disc_params = lib.params_with_name('Discriminator')

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.gen_cost, var_list=gen_params)
                self.disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.disc_cost, var_list=disc_params)


            self.sess.run(tf.initialize_all_variables())
            gen = self.inf_train_gen()

            for iteration in range(self.ITERS):
                start_time = time.time()
                # Train generator
                if iteration > 0:
                    #3.
                    #_ = session.run(gen_train_op)
                    _gen_cost,_ = self.sess.run([self.gen_cost,self.gen_train_op])
                # Train critic
                for i in range(self.CRITIC_ITERS):
                    _data = next(gen)
                        #4.
                        # _disc_cost, _ = session.run(
                        #     [disc_cost, disc_train_op],
                        #     feed_dict={real_inputs_discrete:_data}
                        # )
                    _disc_cost, _ = self.sess.run(
                            [self.disc_cost, self.disc_train_op],
                            feed_dict={self.real_inputs_discrete:_data}
                    )
                        #
                lib.plot.plot('time', time.time() - start_time)
                lib.plot.plot('train disc cost', _disc_cost)

                if iteration % 10 == 9:
                    print("Epoch: [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                            % (iteration,self.ITERS,time.time() - start_time, _disc_cost, _gen_cost))

                if iteration % 100 == 99:
                    #5.
                    #saver.save(session, './my-model', global_step=iteration)
                    #saver.save(sess, self.checkpoint_dir+'/wgan', global_step=iteration)
                    self.save(self.checkpoint_dir, iteration)
                    samples = []
                    for i in range(10):
                        samples.extend(self.generate_samples(self.fake_inputs))

                    for i in range(4):
                        lm = NgramLanguageModel(i+1, samples, tokenize=False)
                        lib.plot.plot('js{}'.format(i+1), lm.js_with(self.true_char_ngram_lms[i]))

                    with open(self.sample_dir+'/samples_{}.txt'.format(iteration), 'w') as f:
                        count=0
                        for s in samples:
                            if set(s).issubset({'A','G','C','T'}):
                                count+=1
                                s = "".join(s)
                                f.write('>'+str(count)+'\n')
                                f.write(s + "\n")

                # if iteration % 100 == 99:
                #     samples = []
                #     for i in range(10):
                #         samples.extend(self.generate_samples(self.fake_inputs))
                            
                #     for i in range(4):
                #         lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                #         lib.plot.plot('js{}'.format(i+1), lm.js_with(self.true_char_ngram_lms[i]))

                #     with open('samples_{}.txt'.format(iteration), 'w') as f:
                #         for i,s in enumerate(samples):
                #             s = "".join(s)
                #             f.write('>' + str(i) + '\n')
                #             f.write(s + "\n")

                    #if iteration % 100 == 99:
                        #lib.plot.flush()
                    
                lib.plot.tick()
        return 


    def save(self, checkpoint_dir, step):
        with open(checkpoint_dir+ '/' + self.model_name + 'charmap.txt','w') as f:
            for c in self.charmap:
                f.write(c+'\t')
                
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)


    def load(self, global_step,checkpoint_dir = None, model_name = None):
        print(" [*] Reading checkpoints...")

        if checkpoint_dir == None:
            checkpoint_dir = self.checkpoint_dir
        if model_name == None:
            model_name = self.model_name
            
        with open(checkpoint_dir+ '/' + model_name + 'charmap.txt','r') as f:
            self.invcharmap = str.split(f.read())
            self.charmap = {}
            i=0
            for c in self.invcharmap:
                self.charmap[c] = i
                i+=1
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #ckpt_name = os.path.basename(ckpt.model_checkpoint_path) #wgan_whc.model-99
            #self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #counter = int(ckpt_name.split('-')[-1])
            load_model_fullname=os.path.join(checkpoint_dir, self.model_name + '.model'+'-'+str(global_step))
            self.saver.restore(self.sess,load_model_fullname)
            print(" [*] Success to read {}".format(load_model_fullname))# ckpt_name
            return True, global_step #counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def generate_final_samples(self,sample_dir,iteration,Nbatch):
        all_samples = []
        for i in range(Nbatch):
            samples = self.sess.run(self.fake_inputs)
            samples = np.argmax(samples, axis=2)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(self.inv_charmap[samples[i][j]])
                if 'u' not in decoded:
                    decoded_samples.append(tuple(decoded))
            all_samples.extend(decoded_samples)
        with open(sample_dir+'/'+'final_samples_{}.txt'.format(iteration), 'w') as f:
            for i,s in enumerate(all_samples):
                s = "".join(s)
                f.write('>' + str(i) + '\n')
                f.write(s + "\n")
        return

    