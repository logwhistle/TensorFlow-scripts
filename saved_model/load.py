import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

pb_file_path = '/Users/liuyuguang/details/tf-scripts/saved_model/savedmodel/'

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], pb_file_path)
    praph = tf.get_default_graph()
    tag_sets = saved_model_utils.get_saved_model_tag_sets(pb_file_path)
    tag_set = ','.join(tag_sets[0])
    meta_graph = saved_model_utils.get_meta_graph_def(pb_file_path, tag_set)
    print(meta_graph.signature_def["signature"].outputs)
    # 生成pbtxt格式文件
    tf.train.write_graph(sess.graph_def, pb_file_path, 'print.pbtxt')