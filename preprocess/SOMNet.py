import tensorflow as tf
import numpy as np
import cv2
sess = tf.InteractiveSession()

def _locations(shape):
    for r in range(shape[0]):
        for c in range(shape[1]):
            yield [r*1.0, c*1.0]

def init(shape = (13,13), dim=3, num_iters=200, learning_rate=0.5):
    '''shape: 输出层节点数
    dim: 权重向量的维数
    '''
    weights = tf.Variable(tf.random_normal([shape[0]*shape[1], dim],mean=0.5, stddev=0.1, dtype="float64"), name="weights", dtype=tf.float64)
    x = tf.placeholder("float64", [1,dim], name='X')
    locations = np.array(list(_locations(shape)))
    locations = tf.constant(locations)
    num_iters = tf.constant([num_iters], dtype=tf.float64)
    learning_rate = tf.constant([learning_rate], dtype=tf.float64)
    radius = tf.constant([shape[0]/2], dtype=tf.float64)
    return {
        'weights':weights,
        'x':x,
        'locations':locations,
        'learning_rate':learning_rate,
        'num_iters':num_iters,
        'radius': radius
    }

def calculate_bmu(weights, x):
    '''找到最佳匹配单元'''
    sub = tf.subtract(weights, x)
    sq = tf.square(sub)
    rs = tf.reduce_sum(sq, -1)
    index = tf.argmin(rs, axis=0)
    return index

def calculate_radius_lrate(iter_step, num_iters, radius0, learning_rate0):
    '''计算半径和学习率
    iter_step: 迭代次数
        num_iters: 总迭代次数
        segma: 初始的'''
    div = tf.negative(tf.divide(iter_step, num_iters))
    radius =  tf.multiply(radius0, tf.exp(div))
    learning_rate = tf.multiply(learning_rate0, tf.exp(div))
    return radius, learning_rate

def neighbors(radius, bmu_index, locations):
    '''radius: 半径
    bmu_index: 激活神经元的索引
    locdations: 表示神经元在输出层的位置的向量
    return: 表示邻域的蒙版向量'''
    slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
#     print('slice:', slice_input.eval())
    slice_input = tf.cast(slice_input, tf.int64)
    bmu = tf.slice(locations, slice_input, [1, 2])
#     print('bmu:', bmu.eval())

    sub = tf.subtract(locations, bmu)
    # 向量距离的平方和
    dist_square = tf.reduce_sum(tf.pow(sub, 2),1)
    # 用一个蒙板表示被选邻域内的节点
    mask = tf.pow(radius, 2) >= dist_square
    return {
        'mask':mask,
        'dist_square':dist_square
    }

def theta(radius, dist_square):
    '''
    radius: 半径
    dist_square: 神经元到BMU的欧几里得距离的平方
    '''
    c = tf.constant(2, dtype=tf.float64)
    div = tf.divide(dist_square, tf.multiply(c, tf.pow(radius, 2)))
    minus = tf.constant(-1, dtype=tf.float64)
    return tf.exp(tf.multiply(minus, div))

def update_weights(weights, mask, xt, lr, theta):
    '''
    weights: 所有神经元的权重矩阵
    mask: 邻域神经元的蒙版
    xt: 输入
    lr: 学习率
    theta: 学习率的权重
    '''
    theta = tf.expand_dims(theta, 1)
    cha = tf.subtract(xt, weights)
    cha = tf.multiply(cha, lr)
    cha = tf.multiply(cha, theta)
    # 把非邻域内的神经元的更新值设置成0 , 这样这些权重就不会更新
    mask = tf.cast(mask, tf.float64)
    mask = tf.expand_dims(mask, 1)
    cha = tf.multiply(cha, mask)
    return tf.add(weights, cha)

def train(colors, num_iters, shape=(13,13),  dim=3, learning_rate=0.5, print_step=40):
    '''训练函数
    colors: 颜色向量
    num_iters: 总训练次数
    shape: 输出层shape
    dim: 输出层向量维度
    learning_rate: 初始学习率'''
    params = init(shape = shape, dim=dim, num_iters=num_iters, learning_rate=learning_rate)
    bmu = calculate_bmu(params['weights'], params['x'])
    iter_step = tf.placeholder(tf.float64, [1], name="iter-step")
    radius, lrate = calculate_radius_lrate(iter_step, params['num_iters'], params['radius'], params['learning_rate'])
    nbs = neighbors(radius, bmu, params['locations'])
    theta_ = theta(radius, nbs['dist_square'])
    new_weights = update_weights(params['weights'], nbs['mask'], params['x'], lrate, theta_)
    train_op = tf.assign(params['weights'], new_weights)
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        
            
        if i == (num_iters-1):
            positions = []
        for j in range(colors.shape[0]):
            x = colors[j].reshape([1,3])
            x.astype('float64')
            radius_, lrate_,bmu_, new_weights_, _ = sess.run([ radius, lrate,bmu,new_weights,train_op], feed_dict={
                params['x']:x,
                iter_step:[i+1]
            })
            if i == (num_iters-1):
                positions.append(bmu_)
                
        if i % print_step == 0 :
            print(i, radius_, lrate_)
    print('Train Finished')
    return {
        'weights':  new_weights_,
        'positions':positions
    }

# from matplotlib import pyplot as plt
# from IPython import display
# def show_image(data, iter_step, radius, lrate):
#     l = int(data.shape[0]**0.5)
#     data = data.reshape([l, l,3])
#     display.clear_output(wait=True)
#     plt.clf()
#     _ = plt.imshow(data, aspect="auto")
#     plt.figtext(0.5, 0.8, f'radius: {radius}', fontsize=15)
#     plt.figtext(0.5, 0.7, f'lrate: {lrate}', fontsize=15)
#     plt.figtext(0.5, 0.6, f'iter_step: {iter_step}', fontsize=15)
#     display.display(plt.gcf())

colors = cv2.imread('.//patch//1702621_0_0_0.jpg').reshape((-1,3))
print(colors.shape)

# from matplotlib import pyplot as plt
data = train(colors, 200, print_step=1, shape=(150,150))
print(data['weights'])
print(data['positions'])