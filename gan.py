#!/usr/bin/python
#coding:utf-8
import os, time, pickle, random, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model_gan import GAN_g1, GAN_g2, GAN_g, GAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
import test_data
from PIL import Image


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init1 = config.TRAIN.lr_init1
lr_init2 = config.TRAIN.lr_init2
beta1 = config.TRAIN.beta1
alpha = config.TRAIN.alpha
beta = config.TRAIN.beta
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():

    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

  ###====================== PRE-LOAD DATA ===========================###

    file = open('2007_train.txt', 'r')
    img_paths = []
    corlabs = []
    for c in file.readlines():
        c_array = c.split(" ")
        img_paths.append(c_array[0])
        # img = Image.open(img_paths)
        for i in range(1, len(c_array)):
            corlabs.append(c_array[i])



    # train_hr_img_list = sorted(tl.files.load_file_list(path=img_paths, printable=False))

    train_hr_imgs = tl.vis.read_images(img_paths)







    lab = tl.files.read_file(config.TRAIN.hr_lab_path)
    label_str = lab.split(' ')



    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 24, 24, 3], name='t_image_input')
    t_target_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_target_image')
    label = tf.placeholder('int32', [batch_size, ], name='label') #??????

    net_g1 = GAN_g1(t_image, is_train=True, reuse=False)
    net_g = GAN_g(t_image, is_train=True, reuse=False)

    net_d1, logits_real1, _, logits_real2 = GAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake1, _, logits_fake2 = GAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_g.print_layers()
    net_g1.print_params(False)
    net_g1.print_layers()
    net_d1.print_params(False)
    net_d1.print_layers()
    print('sssssssss')


   # vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    t_predict_image = net_g.outputs
    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = GAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real1, tf.ones_like(logits_real1), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake1, tf.zeros_like(logits_fake1), name='d2')
    d_loss3 = tl.cost.cross_entropy(logits_fake2, label, name='d3') #?????
    d_loss4 = tl.cost.cross_entropy(logits_real2, label, name='d4')  # ?????
    d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4

    adv_loss = alpha * tl.cost.sigmoid_cross_entropy(logits_fake1, tf.zeros_like(logits_fake1), name='adv')  #???????
    mse_loss1 = tl.cost.mean_squared_error(net_g1.outputs, t_target_image, is_mean=True)
    mse_loss2 = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    mse_loss = mse_loss1 + mse_loss2
    clc_loss = beta * tl.cost.cross_entropy(logits_fake2, label, name='clc')
    g_loss = mse_loss + clc_loss + adv_loss

    g_vars = tl.layers.get_variables_with_name('GAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('GAN_d', True, True)

    # with tf.variable_scope('learning_rate'):
    #     if n_epoch <= 3:
    #         lr_v = tf.Variable(lr_init1, trainable=False)
    #     else:
    #         lr_v = tf.Variable(lr_init2, trainable=False)

          # ????????????????????????

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init1, trainable=False)

 ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)

    ## GAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    print('aaaaaaaaaaaaa')

 ###========================== RESTORE MODEL =============================###
    config1 = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))
    # config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    writer = tf.summary.FileWriter("logs_gan/", sess.graph)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d1)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs, fn=get_imgs)
    # sample_imgs_384 = [sample_imgs[i] for i in batch_size]
    # sample_imgs_hr = sample_imgs
    # sample_imgs_lr = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample HR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    #(16, 384, 384, 3) -1.0 1.0
    sample_imgs_24 = tl.prepro.threading_data(sample_imgs_96, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_24.shape, sample_imgs_24.min(), sample_imgs_24.max())
    #(16, 96, 96, 3) -1.0 1.0
    print(len(sample_imgs_96))
    print(len(sample_imgs_24))
    print(ni)
    tl.vis.save_images(sample_imgs_24, [ni, ni], save_dir_ginit + '/_train_sample_lr.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_hr.png')
    tl.vis.save_images(sample_imgs_24, [ni, ni], save_dir_gan + '/_train_sample_lr.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_hr.png')

 ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init1))
    print(" ** fixed learning rate: %f (for init G)" % lr_init1)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0


        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            # b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn,
            #                                       is_random=True)
            # b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            b_imgs_hr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=get_imgs)
            # b_imgs_hr = train_hr_imgs[idx:idx + batch_size]
            b_imgs_lr = tl.prepro.threading_data(b_imgs_hr, fn=downsample_fn)
            ## update G
            errM1, _ = sess.run([mse_loss1, g_optim_init], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f, mse1: %.8f " % (
            epoch, n_epoch_init, n_iter, time.time() - step_time, errM, errM1))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
        epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            out = sess.run(net_g_test.outputs,
                           {t_image: sample_imgs_24})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 2 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                              sess=sess)

        ###========================= train GAN  =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch <= 3:
            sess.run(tf.assign(lr_v, lr_init1))
            log = " ** learning rate: %f (for GAN)" % (lr_init1)
            print(log)
        else:
            sess.run(tf.assign(lr_v, lr_init2))
            log = " ** new learning rate: %f (for GAN)" % (lr_init2)
            print(log)
        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_hr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=get_imgs)
            b_imgs_lr = tl.prepro.threading_data(b_imgs_hr, fn=downsample_fn)
            b_label_str = label_str[idx:idx + batch_size]
            b_labels = [int(x) for x in b_label_str]

            # b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn,
            #                                       is_random=True)
            # b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr, label: b_labels})
            ## update G
            errG, errM, errC, errA, _ = sess.run([g_loss, mse_loss, clc_loss, adv_loss, g_optim],
                                                 {t_image: b_imgs_lr, t_target_image: b_imgs_hr, label: b_labels})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f clc: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errC, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
        epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
        total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            out = sess.run(net_g_test.outputs,
                           {t_image: sample_imgs_24})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 2 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                              sess=sess)
            tl.files.save_npz(net_d1.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
                              sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.jpg', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 2  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    # valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    print(valid_lr_img)

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = GAN_g(t_image, is_train=False, reuse=False)
    print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_train.npz', network=net_g)
    print('wwwwwwwwwwwwwwwwwwwwwwww')
    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    # print(out[0])
    tl.vis.save_image(out[0], save_dir + '/valid_gen.jpg')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.jpg')
    # tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.jpg')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode


    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")


