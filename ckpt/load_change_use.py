from tensorflow.python.training import checkpoint_utils as cp
import tensorflow as tf
import numpy as np

# 方法一：加载并打印checkpoint, 如需要改变原有checkpoint文件则将'need_change=True'(此处以增添维度为例，输入增加8维时，方便模型后续热启动训练), glort函数为常用随机初始化方法
def glort(input, output):
    limit = np.sqrt(6/(input + output))
    return np.random.uniform(-limit, limit, size=[input, output])

def load_and_change(need_change=False):
    with tf.Session() as sess:
        path = './checkpoint/model'
        source_reader = cp.load_checkpoint(path)
        for name, shape in cp.list_variables(path):
            v = source_reader.get_tensor(name)
            print(shape, ' ', name, ' : ', v)

            if need_change:
                # 此处重新定义新变量var可以用于构造新的图并使用，或者保存为新的checkpoint文件
                if name == 'models/weight':
                    v1 = np.concatenate([v, glort(8, 16)], 0)
                    var = tf.Variable(v1, name=name, dtype=tf.float32)
                    # print(shape, ' ', name)
                else:
                    var = tf.Variable(v, name=name)
        
        if need_change:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            vars = tf.global_variables()
            for var in vars:
                print('new_chpt ', var.name, sess.run(tf.shape(var)))
            saver.save(sess, './new_checkpoint/model')

# 使用加载及改变checkpoint的函数
# load_and_change(True)


# 方法二：加载checkpoint并重新使用
def load_and_reuse():
    with tf.Session() as sess:
        # 占位符placeholder不在checkpoint中，需要重新定义
        inputs = tf.placeholder(tf.float32, name='input')
        label1 = tf.placeholder(tf.float32, name='label1')
        label2 = tf.placeholder(tf.float32, name='label2')

        # 通过meta文件加载图, 通过checkpoint文件加载变量
        saver = tf.train.import_meta_graph('./checkpoint/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/'))

        graph = tf.get_default_graph()
        # 变量加载，同时需要重新定义模型
        w0 = graph.get_tensor_by_name('models/weight:0')
        b0 = graph.get_tensor_by_name('models/bias:0')
        h1 = tf.nn.elu(tf.add(tf.tensordot(inputs, w0, axes=1), b0))

        # MTL
        w_l1 = graph.get_tensor_by_name('models/weight_label1:0')
        b_l1 = graph.get_tensor_by_name('models/bias_label1:0')
        y1 = tf.add(tf.tensordot(h1, w_l1, axes=1), b_l1)

        w_l2 = graph.get_tensor_by_name('models/weight_label2:0')
        b_l2 = graph.get_tensor_by_name('models/bias_label2:0')
        y2 = tf.add(tf.tensordot(h1, w_l2, axes=1), b_l2)

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label1, logits=y1, name='l1_loss') + 
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label2, logits=y2, name='l2_loss')
            )
        
        
        step = cp.load_checkpoint('./checkpoint/model').get_tensor('global_step')
        global_step = tf.Variable(step, trainable=False, name='global_step')
        opt = tf.train.GradientDescentOptimizer(0.01)  # 随机梯度下降
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='models/')  # model范围在的所有变量都更新，也可以在此处手动跳过某些变量
        # 如无需对某些参数跳过，或对梯度额外操作（如梯度剪裁, GradNorm, PCGrad等）, 可将compute_gradients及apply_gradients合并为opt.minimize()
        grad_var = opt.compute_gradients(loss, var_list=tvars)
        train_op = opt.apply_gradients(grad_var, global_step=global_step)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        input = 3 * np.random.random([32, 32]) + 2
        input = input.astype(np.float32)
        l1 = np.round(1 / (1 + np.mean(-input * input, axis=1, keepdims=True)))
        l1 = l1.astype(np.float32)
        l2 = np.round(1 / (1 + np.mean(-2 * input ** 2 + 0.5, axis=1, keepdims=True)))
        l2 = l2.astype(np.float32)

        # print(sess.run(global_step))

        for i in range(5):
            _, l, step = sess.run([train_op, loss, global_step], feed_dict={inputs: input, label1: l1, label2: l2})
            print(l, step)

load_and_reuse()
        