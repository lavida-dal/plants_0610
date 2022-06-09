import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow import keras
# from tensorflow.python.platform import gfile
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
# from tensorflow_examples.lite.model_maker.core.task import image_classifier


model = tf.keras.models.load_model("./save_model/h5/pa.0526.50.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfLite_model = converter.convert()
open("converted_model_pa_0526_30.tfLite","wb").write(tfLite_model)

# data = ImageClassifierDataLoader.from_folder('folder/')
# train_data, test_data = data.split(0.8)
# model = image_classifier.create(train_data)
# loss, accuracy = model.evaluate(test_data)
# model.export('gochu.tfLite', 'gochulabels.pbtxt')

# model = keras.models.load_model("./gochu_weights.h5", compile=True)
#
# export_path = './save_model_old/'
# model.save(export_path, save_format='tf')

# def convert_pb_to_pbtxt(filename):
#
# with gfile.FastGFile(filename,'rb') as f:
#
# graph_def = tf.GraphDef()
#
# graph_def.ParseFromString(f.read())
#
# tf.import_graph_def(graph_def, name='')
#
# tf.train.write_graph(graph_def,'./','protobuf.pbtxt', as_text=True)
#
# return