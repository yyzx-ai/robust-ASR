import functools
import numpy as np
import imlib as im
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import matplotlib.pyplot as plt
import data
import module
import os
import shutil
import scipy.io.wavfile as wav1
from python_speech_features import mfcc
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
path='/home/dell/文档/语音识别代码实例/speech_recognition_cnn_gan-master/speech_commands/data/'
dir_list=os.listdir(path)
#name_='down'

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line
py.arg('--dataset', default='self', choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'anime', 'custom','self'])
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=40000)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--n_g', type=int, default=5)
py.arg('--z_dim', type=int, default=128)
py.arg('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='wgan-div', choices=['none', 'dragan', 'wgan-gp','wgan-div'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--experiment_name', default='none')
args = py.args()
def main_(name_):
    #fs,data11=wav1.read('sheli.wav')
    #feature=mfcc(data11, samplerate=fs,numcep=32,nfilt=64,winfunc=np.hamming)
    # output_dir
    if args.experiment_name == 'none':
        args.experiment_name = '%s_%s' % (args.dataset, args.adversarial_loss_mode)
        if args.gradient_penalty_mode != 'none':
            args.experiment_name += '_%s' % args.gradient_penalty_mode
    output_dir = py.join('output', args.experiment_name)
    #print('-----------------------------------\n\n\n%s'%args.experiment_name)
    #print('-----------------------------------\n\n\n%s'%output_dir)
    py.mkdir(output_dir)

    # save settings
    py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================

    # setup dataset
    if args.dataset in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
        dataset, shape, len_dataset = data.make_32x32_dataset(args.dataset, args.batch_size)
        n_G_upsamplings = n_D_downsamplings = 3

    elif args.dataset == 'self':
        dataset, shape, len_dataset = data.make_self_dataset(path+name_,args.batch_size)
        n_G_upsamplings = n_D_downsamplings = 3

    elif args.dataset == 'celeba':  # 64x64
        img_paths = py.glob('data/img_align_celeba', '*.jpg')
        dataset, shape, len_dataset = data.make_celeba_dataset(img_paths, args.batch_size)
        n_G_upsamplings = n_D_downsamplings = 4

    elif args.dataset == 'anime':  # 64x64
        img_paths = py.glob('data/faces', '*.jpg')
        dataset, shape, len_dataset = data.make_anime_dataset(img_paths, args.batch_size)
        n_G_upsamplings = n_D_downsamplings = 4

    elif args.dataset == 'custom':
        # ======================================
        # =               custom               =
        # ======================================
        img_paths = ...  # image paths of custom dataset
        dataset, shape, len_dataset = data.make_custom_dataset(img_paths, args.batch_size)
        n_G_upsamplings = n_D_downsamplings = ...  # 3 for 32x32 and 4 for 64x64
        # ======================================
        # =               custom               =
        # ======================================


    # ==============================================================================
    # =                                   model                                    =
    # ==============================================================================

    # setup the normalization function for discriminator
    if args.gradient_penalty_mode == 'none':
        d_norm = 'batch_norm'
    elif args.gradient_penalty_mode in ['dragan', 'wgan-gp','wgan-div']:  # cannot use batch normalization with gradient penalty
        # TODO(Lynn)
        # Layer normalization is more stable than instance normalization here,
        # but instance normalization works in other implementations.
        # Please tell me if you find out the cause.
        d_norm = 'layer_norm'

    # networks
    G = module.ConvGenerator( output_channels=shape[-1], n_upsamplings=n_G_upsamplings, name='G_%s' % args.dataset)
    D = module.ConvDiscriminator(input_shape=shape, n_downsamplings=n_D_downsamplings, norm=d_norm, name='D_%s' % args.dataset)
    #G.summary()
    #D.summary()

    # adversarial_loss_functions
    d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

    G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)


    # ==============================================================================
    # =                                 train step                                 =
    # ==============================================================================

    @tf.function
    def train_G():
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
            x_fake = G(z, training=True)
            x_fake_d_logit = D(x_fake, training=True)
            G_loss = g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

        return {'g_loss': G_loss}


    @tf.function
    def train_D(x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
            x_fake = G(z, training=True)

            x_real_d_logit = D(x_real, training=True)
            x_fake_d_logit = D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
            #print('\n---{}\n---{}\n'.format(x_real, x_fake))
            gp = gan.gradient_penalty(functools.partial(D, training=True), x_real, tf.cast(x_fake,dtype=tf.float64), mode=args.gradient_penalty_mode)
            #print('\n---{}\n---{}\n---{}\n'.format(x_real_d_loss, x_fake_d_loss,gp))
            D_loss = (tf.cast(x_real_d_loss,dtype=tf.float64) + tf.cast(x_fake_d_loss,dtype=tf.float64)) + gp * args.gradient_penalty_weight

        D_grad = t.gradient(D_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


    @tf.function
    def sample(z):
        return G(z, training=False)

    def copydirs(from_file, to_file):
        if not os.path.exists(to_file):  # 如不存在目标目录则创建
            os.makedirs(to_file)
        files = os.listdir(from_file)  # 获取文件夹中文件和目录列表
        for f in files:
            if os.path.isdir(from_file + '/' + f):  # 判断是否是文件夹
                copydirs(from_file + '/' + f, to_file + '/' + f)  # 递归调用本函数
            else:
                shutil.copy(from_file + '/' + f, to_file + '/' + f)  # 拷贝文件

    # ==============================================================================
    # =                                    run                                     =
    # ==============================================================================

    # epoch counter
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # checkpoint
    checkpoint = tl.Checkpoint(dict(G=G,
                                    D=D,
                                    G_optimizer=G_optimizer,
                                    D_optimizer=D_optimizer,
                                    ep_cnt=ep_cnt),
                               py.join(output_dir, 'checkpoints'),
                               max_to_keep=5)
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().assert_existing_objects_matched()
    except Exception as e:
        print(e)

    # summary
    train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

    # sample
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)
    #from tensorflow.keras.utils import plot_model
    #plot_model(G, to_file='G_model.png', show_shapes=True,rankdir='TB',show_layer_names=True)
    #plot_model(D, to_file='D_model.png', show_shapes=True,rankdir='TB',show_layer_names=True)
    # main loop
    z = tf.random.normal((args.batch_size, 1, 1, args.z_dim))  # a fixed noise for sampling
    d_los=0
    g_los=0
    g_list=[]
    d_list=[]
    d_loss_list=[]
    g_loss_list=[]
    gp_loss=[]
    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue
            # update epoch counter
            ep_cnt.assign_add(1)
            # train for an epoch
            for x_real in tqdm.tqdm(dataset, desc='Inner Epoch Loop', total=len_dataset):
                G_loss_dict = train_G()
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

                g_list.append(float(G_loss_dict.get('g_loss')))
                if G_optimizer.iterations.numpy() % args.n_g == 0:
                    D_loss_dict = train_D(x_real)
                    tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')

                    d_list.append(float(D_loss_dict.get('d_loss')))
            if ep%100==0:
                #plt.subplot(1,2,1)
                #plt.imshow(feature.T,origin='lower')
                #plt.subplot(1,2,2)
                x_fake_ = sample(tf.random.normal((1, 1, 1, args.z_dim)))
                plt.imshow(np.squeeze((255*(x_fake_.numpy())).T),origin='lower')
                plt.savefig('{}/samples_training/{}.png'.format(output_dir,ep))
                plt.close()
                np.save('output/d_list.npy'.format(args.experiment_name,name_,),np.array(d_list))
                np.save('output/g_list.npy'.format(args.experiment_name,name_,),np.array(g_list))
            #if ep%10==0:
                #x_fake = sample(tf.random.normal((100, 1, 1, args.z_dim)))
                #img = im.immerge(x_fake, n_rows=10).squeeze()
                #im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
            gp_loss.append(float(D_loss_dict.get('gp')))
            checkpoint.save(ep)

    #for i3 in range(1000):
        #if not os.path.exists('gen/%s'%name_):  # 如不存在目标目录则创建
            #os.makedirs('gen/%s'%name_)
        
        #x_fake_1 = sample(tf.random.normal((1, 1, 1, args.z_dim)))
        #x_fake_2=255*x_fake_1.numpy()
        #x_fake_2=np.squeeze(x_fake_2)
        #np.save('gen/{}/{}_{}.npy'.format(name_,name_,i3),x_fake_2)
    #-----保存模型到另一个文件夹---
    #os.mkdir('output/detail/self_wgan_wgan-gp_{}'.format(name_))
    copydirs('{}'.format(output_dir),'output/detail/{}_{}'.format(args.experiment_name,name_))
    #os.rmdir('output/self_wgan_wgan-gp/')
    #-----可视化---
    #print('D_loss:{}\nG_loss:{}'.format(d_loss,g_loss))
    np.save('output/{}_{}/d_list.npy'.format(args.experiment_name,name_,),np.array(d_list))
    np.save('output/{}_{}/g_list.npy'.format(args.experiment_name,name_,),np.array(g_list))
    plt.subplot(1,3,1)
    plt.plot(g_list)
    #plt.plot(g_loss_list)
    plt.xlabel('g_loss')
    #--------------
    plt.subplot(1,3,2)
    plt.plot(d_list)
    #plt.plot(d_loss_list)
    plt.xlabel('d_loss')
    plt.subplot(1,3,3)
    plt.plot(gp_loss)
    plt.xlabel('gp_loss')
    plt.savefig('output/detail/{}_{}/{}.png'.format(args.experiment_name,name_,name_))
    plt.close()
    #plt.show()
    shutil.rmtree(output_dir)



#start:
for file_ in tqdm.tqdm(dir_list[8:9]):
    print(file_)
    main_(file_)



















