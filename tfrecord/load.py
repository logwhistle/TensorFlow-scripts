import tensorflow as tf
# 解决“sess is empty"问题，使用sess调用迭代器打印
tf.compat.v1.disable_eager_execution()
# 简单打印
# tf.enable_eager_execution()

def read_record():
    raw_dataset = tf.data.TFRecordDataset('/Users/liuyuguang/details/tf-scripts/tfrecord/my.tfrecords')

    # FixedLenFeature是dense特征，VarLenFeature是变长的sparse特征
    feature_description = {
        'label': tf.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'features': tf.FixedLenFeature([2], tf.float32, default_value=[0.0, 0.0]),
        'sparse': tf.VarLenFeature(tf.float32),
    }
    
    # Define the parse function to extract a single example as a dict.
    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, feature_description)

    sess = tf.InteractiveSession()
    parsed_dataset = raw_dataset.map(_parse_function).batch(2)
    iterator = parsed_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # 简单打印
    # for image_features in parsed_dataset:
    #     image_raw = image_features['image_raw'].numpy()
    #     label = image_features['label'].numpy()
    #     print(label)

    # 解决“sess is empty"问题，使用sess调用迭代器打印
    while True:
        try:
            raw_features = sess.run(next_element)
            # print(raw_features.keys())
            image_raw = raw_features['image_raw']
            label = raw_features['label']
            features = raw_features['features']
            sparse = raw_features['sparse']
            print('label: ', label, '\nfeatures: ', features, '\nsparse: ', sparse, '\n')
        except:
            print('end')
            break

        
if __name__ == "__main__":
    read_record()