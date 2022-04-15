import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = tf.train.latest_checkpoint('../model/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='', all_tensor_names=True)