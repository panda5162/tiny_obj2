
import time
import tensorflow as tf
from tensorlayer.layers import *
import cv2


def GAN_g1(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("GAN_g1", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = DeConv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/dc/m')
        n = DeConv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s3/dc/m')
        n = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='n3s1/c')

        return(n)


def GAN_g2(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("GAN_g2", reuse=reuse) as vs:
        # n = InputLayer(t_image, name='in')
        n = Conv2d(t_image, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c/m')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c/m')
        n = Conv2d(n, 3, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='n3s1/c')

        return(n)


def GAN_g(t_image, is_train=False, reuse=False):
    with tf.variable_scope("GAN_g", reuse=reuse) as vs:
        n = GAN_g1(t_image, is_train, reuse)
        n = GAN_g2(n, is_train, reuse)
        return n


# def conv2d(t_image, filters, shape, stride = ( 1, 1 ), training=True):
#     layer = tf.layers.conv2d( t_image,
#                               filters,
#                               shape,
#                               stride,
#                               padding = 'SAME',
#                               kernel_initializer=tf.truncated_normal_initializer( stddev=0.01 ) )
#     layer = tf.layers.batch_normalization( layer, training = training )
#     layer = Leaky_Relu( layer )
#     return layer
#
#
# def Leaky_Relu(t_image, alpha = 0.01 ):
#     output = tf.maximum( t_image, tf.multiply( t_image, alpha ) )
#     return output
#
#
# def Res_conv2d( t_image, shortcut, filters, shape, stride = ( 1, 1 ), training =True):
#     conv = conv2d( t_image, filters, shape, training = training )
#     Res = Leaky_Relu( conv + shortcut )
#     return Res
#
#
# def feature_extractor( t_image, is_train):
#     layer = conv2d( t_image, 32, [3, 3], training = is_train)
#     layer = conv2d( layer, 64, [3, 3], ( 2, 2 ), training = is_train)
#     shortcut = layer
#
#     layer = conv2d( layer, 32, [1, 1], training = is_train )
#     layer = Res_conv2d( layer, shortcut, 64, [3, 3], training = is_train )
#
#     layer = conv2d( layer, 128, [3, 3], ( 2, 2 ), training = is_train )
#     shortcut = layer
#
#     for _ in range( 2 ):
#         layer = conv2d( layer, 64, [1, 1], training = is_train )
#         layer = Res_conv2d( layer, shortcut, 128, [3, 3], training = is_train )
#
#     layer = conv2d( layer, 256, [3, 3], ( 2, 2 ), training = is_train )
#     shortcut = layer
#
#     for _ in range( 8 ):
#         layer = conv2d( layer, 128, [1, 1], training = is_train )
#         layer = Res_conv2d( layer, shortcut, 256, [3, 3], training = is_train )
#     pre_scale3 = layer
#
#     layer = conv2d( layer, 512, [3, 3], ( 2, 2 ), training = is_train )
#     shortcut = layer
#
#     for _ in range( 8 ):
#         layer = conv2d( layer, 256, [1, 1], training = is_train )
#         layer = Res_conv2d( layer, shortcut, 512, [3, 3], training = is_train )
#     pre_scale2 = layer
#
#     layer = conv2d( layer, 1024, [3, 3], ( 2, 2 ), training = is_train )
#     shortcut = layer
#
#     for _ in range( 4 ):
#         layer = conv2d( layer, 512, [1, 1], training = is_train )
#         layer = Res_conv2d( layer, shortcut, 1024, [3, 3], training = is_train )
#     pre_scale1 = layer
#     return pre_scale1, pre_scale2, pre_scale3
#
#
# def get_layer2x( layer_final, pre_scale ):
#     layer2x = tf.image.resize_images(layer_final,
#                                      [2 * tf.shape(layer_final)[1], 2 * tf.shape(layer_final)[2]])
#     layer2x_add = tf.concat( [layer2x, pre_scale], 3 )
#     return layer2x_add
#
#
# def scales( layer, pre_scale2, pre_scale3, is_train):
#     layer = conv2d(layer, 512, [1, 1], training = is_train)
#     layer = conv2d(layer, 1024, [3, 3], training = is_train)
#     layer = conv2d(layer, 512, [1, 1], training = is_train)
#     layer_final = layer
#     layer = conv2d(layer, 1024, [3, 3], training = is_train)
#
#     '''--------scale_1--------'''
#     scale_1 = conv2d(layer, 255, [1, 1], training = is_train)
#
#     '''--------scale_2--------'''
#     layer = conv2d(layer_final, 256, [1, 1], training = is_train)
#     layer = get_layer2x(layer, pre_scale2 )
#
#     layer = conv2d(layer, 256, [1, 1], training = is_train)
#     layer= conv2d(layer, 512, [3, 3], training = is_train)
#     layer = conv2d(layer, 256, [1, 1], training = is_train)
#     layer = conv2d(layer, 512, [3, 3], training = is_train)
#     layer = conv2d(layer, 256, [1, 1], training = is_train)
#     layer_final = layer
#     layer = conv2d(layer, 512, [3, 3], training = is_train)
#     scale_2 = conv2d(layer, 255, [1, 1], training = is_train)
#
#     '''--------scale_3--------'''
#     layer = conv2d(layer_final, 128, [1, 1], training = is_train)
#     layer = get_layer2x(layer, pre_scale3 )
#
#     for _ in range( 3 ):
#         layer = conv2d(layer, 128, [1, 1], training = is_train)
#         layer = conv2d(layer, 256, [3, 3], training = is_train)
#     scale_3 = conv2d(layer, 255, [1, 1], training = is_train)
#
#     scale_1 = tf.abs( scale_1 )
#     scale_2 = tf.abs( scale_2 )
#     scale_3 = tf.abs( scale_3 )
#
#     return scale_1, scale_2, scale_3
#
#
#
# '''--------Test the scale--------'''
# if __name__ == "__main__":
#     data = cv2.imread('VOCROOT/VOC2007/JPEGImages/000005.jpg')
#     data = cv2.cvtColor( data, cv2.COLOR_BGR2RGB )
#     data = cv2.resize( data, ( 416, 416 ) )
#
#     data = tf.cast( tf.expand_dims( tf.constant( data ), 0 ), tf.float32 )
#
#     pre_scale1, pre_scale2, pre_scale3 = feature_extractor(data, is_train=False)
#
#     scale_1, scale_2, scale_3 = scales( pre_scale1, pre_scale2, pre_scale3, is_train=False)
#
#     with tf.Session() as sess:
#
#         sess.run( tf.initialize_all_variables() )
#
#         print( sess.run( scale_1 ).shape )




# def GAN_d(t_image, is_train=False, reuse=False):
#
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     b_init = None
#     g_init = tf.random_normal_initializer(1., 0.02)
#     with tf.variable_scope("GAN_d", reuse=reuse) as vs:
#         # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
#         n = InputLayer(t_image, name='in')
#         conv1_1 = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv1_1')
#         conv1_2 = Conv2d(conv1_1, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv1_2')
#         pool1 = MaxPool2d(conv1_2, (2, 2), (2, 2), name='pool1')
#
#         conv2_1 = Conv2d(pool1, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv2_1')
#         conv2_2 = Conv2d(conv2_1, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv2_2')
#         pool2 = MaxPool2d(conv2_2, (2, 2), (2, 2), name='pool2')
#
#         conv3_1 = Conv2d(pool2, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv3_1')
#         conv3_2 = Conv2d(conv3_1, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv3_2')
#         conv3_3 = Conv2d(conv3_2, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv3_3')
#         conv3_4 = Conv2d(conv3_3, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv3_4')
#         pool3 = MaxPool2d(conv3_4, (2, 2), (2, 2), name='pool3')
#
#         conv4_1 = Conv2d(pool3, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv4_1')
#         conv4_2 = Conv2d(conv4_1, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv4_2')
#         conv4_3 = Conv2d(conv4_2, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv4_3')
#         conv4_4 = Conv2d(conv4_3, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='conv4_4')
#         pool4 = MaxPool2d(conv4_4, (2, 2), (2, 2), name='pool4')
#
#         conv5_1 = Conv2d(pool4, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv5_1')
#         conv5_2 = Conv2d(conv5_1, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv5_2')
#         conv5_3 = Conv2d(conv5_2, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv5_3')
#         conv5_4 = Conv2d(conv5_3, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv5_4')
#
#
#         n = FlattenLayer(conv5_4, name='f')
#         n1 = DenseLayer(n, n_units=2, act=tf.identity, name='out1')
#         n2 = DenseLayer(n, n_units=30, act=tf.identity, name='out2')
#
#         logits1 = n1.outputs
#         #n1.outputs = tf.nn.sigmoid(n1.outputs)
#
#         logits2 = n2.outputs
#         #n2.outputs = tf.nn.softmax(n2.outputs)
#         return n1, logits1, n2, logits2


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv


