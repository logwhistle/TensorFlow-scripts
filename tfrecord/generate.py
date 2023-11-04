import tensorflow as tf
# 解决“sess is empty"问题
tf.compat.v1.disable_eager_execution()


# All raw values should be converted to a type compatible with tf.Example. 
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(label, image_string, features, sparse):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
        'features': _float_feature(features),
        'sparse': _float_feature(sparse),
    }
    # 调用api将example序列化字节字符串
    example_prote = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_prote.SerializeToString()

def write_record():
    image_string = [open('/Users/liuyuguang/details/tf-scripts/pictures/MTL.svg', 'rb').read()]
    label = [0, 1, 2]
    features = [0.1, 0.2]

    with tf.python_io.TFRecordWriter('/Users/liuyuguang/details/tf-scripts/tfrecord/my.tfrecords') as writer:
        writer.write(serialize_example(label, image_string, features, [1, 2, 3]))
        writer.write(serialize_example(label, image_string, features, [4, 5]))
        writer.write(serialize_example(label, image_string, features, [6, 7, 8, 9]))
        writer.write(serialize_example(label, image_string, features, [10]))


        
if __name__ == "__main__":
    write_record()