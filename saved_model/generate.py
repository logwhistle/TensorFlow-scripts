import tensorflow as tf
import numpy as np

np.random.seed(0)  # 设置随机种子
hidden_size = 16  # 设置隐藏层维度

# 一个简单的多任务网络，常用于推荐网络中
with tf.variable_scope('models', reuse=tf.AUTO_REUSE) as scope:
    # 设置占位符placeholder，后续通过feed_dict传入. 注意传入时的维度和数据类型统一，否则会报错
    inputs = tf.placeholder(tf.float32, name='input')
    label1 = tf.placeholder(tf.float32, name='label1')
    label2 = tf.placeholder(tf.float32, name='label2')

    # 第一层网络
    w0 = tf.get_variable(name='weight', initializer=tf.variance_scaling_initializer, shape=[32, hidden_size])
    b0 = tf.get_variable(name='bias', initializer=tf.zeros_initializer(), shape=[hidden_size])
    h1 = tf.nn.elu(tf.add(tf.tensordot(inputs, w0, axes=1), b0))

    # MTL
    w_l1 = tf.get_variable('weight_label1', initializer=tf.variance_scaling_initializer, shape=[hidden_size, 1])
    b_l1 = tf.get_variable(name='bias_label1', initializer=tf.zeros_initializer(), shape=[1])
    y1 = tf.add(tf.tensordot(h1, w_l1, axes=1), b_l1)

    w_l2 = tf.get_variable('weight_label2', initializer=tf.variance_scaling_initializer, shape=[hidden_size, 1])
    b_l2 = tf.get_variable(name='bias_label2', initializer=tf.zeros_initializer(), shape=[1])
    y2 = tf.add(tf.tensordot(h1, w_l2, axes=1), b_l2)

    # 两个任务的loss加权求和：此时y1和y2不需要额外经过sigmoid激活函数，“sigmoid_cross_entropy_with_logits”中会自动过sigmoid
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label1, logits=y1, name='l1_loss') + 
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label2, logits=y2, name='l2_loss')
        )

global_step = tf.Variable(0, trainable=False, name='global_step')
opt = tf.train.GradientDescentOptimizer(0.01)  # 随机梯度下降
tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='models/')  # model范围在的所有变量都更新，也可以在此处手动跳过某些变量
# 如无需对某些参数跳过，或对梯度额外操作（如梯度剪裁, GradNorm, PCGrad等）, 可将compute_gradients及apply_gradients合并为opt.minimize()
grad_var = opt.compute_gradients(loss, var_list=tvars)
train_op = opt.apply_gradients(grad_var, global_step=global_step)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 构造数据集，此处不能与上述placeholder重名，且对应传入的类型和shape需要一致
input = 3 * np.random.random([32, 32]) + 2
input = input.astype(np.float32)
l1 = np.round(1 / (1 + np.mean(-input * input, axis=1, keepdims=True)))
l1 = l1.astype(np.float32)
l2 = np.round(1 / (1 + np.mean(-2 * input ** 2 + 0.5, axis=1, keepdims=True)))
l2 = l2.astype(np.float32)

for i in range(5):
    _, l, step = sess.run([train_op, loss, global_step], feed_dict={inputs: input, label1: l1, label2: l2})
    print(l)

# 以上MLP demo同ckpt文件夹中的generate.py
'''
生成savedmodel文件，文件包含三部分
variables文件夹
pb文件为固化的图结构
tf_serving_warmup_request文件是便于serving热启动的文件
'''

from tensorflow.python.saved_model import tag_constants

save_input, save_output = {}, {}
# 生成signature文件
save_input["inputs"] = tf.saved_model.utils.build_tensor_info(inputs)
save_output["y1"] = tf.saved_model.utils.build_tensor_info(y1)
save_output["y2"] = tf.saved_model.utils.build_tensor_info(y2)
signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=save_input,
    outputs=save_output,
    method_name=tf.saved_model.signature_constants.PREDICT_INPUTS
)

# 图文件
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="saved")

# 此处生成了savedmodel中的variables文件夹 和 pb文件
builder = tf.saved_model.builder.SavedModelBuilder('/Users/liuyuguang/details/tf-scripts/saved_model/savedmodel')
with tf.Session(graph=graph) as sess:
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'signature':signature})
    builder.save()


# 生成tf_serving_warmup_request文件
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

examples = {}
examples["inputs"] = tf.make_tensor_proto([0]*32, dtype=tf.float32)
local_path = '/Users/liuyuguang/details/tf-scripts/saved_model/savedmodel/tf_serving_warmup_request'
with tf.python_io.TFRecordWriter(local_path) as writer:
    for i in range(1):
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(
                name='mlp_demo',
                signature_name='signature'
            ),
            inputs=examples,
            output_filter=["y1", "y2"]
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )
        writer.write(log.SerializeToString())