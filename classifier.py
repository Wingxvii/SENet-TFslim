import os

import tensorflow as tf
import os
import numpy as np
import cv2
from preprocessing import preprocessing_factory


def main(_):

    graph_def = tf.compat.v1.GraphDef()
    labels = ['map', 'game', 'dialogue']

    # These are set to the default names from exported models, update as needed.
    filename = "model.pb"
    labels_filename = "labels.txt"

    # Import the TF graph
    with tf.io.gfile.GFile('C:/Users/turnt/OneDrive/Desktop/Rob0 Workspace/Scene_labeler/data/graph.pbtxt', 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # Create a list of labels.
    with open('C:/Users/turnt/OneDrive/Desktop/Rob0 Workspace/Scene_labeler/classes.txt', 'rt') as lf:
      for l in lf:
        labels.append(l.strip())

    # Load from a file
    image = Image.open('C:/Users/turnt/OneDrive/Desktop/Rob0 Workspace/Scene_labeler/input_images/test/001.jpg')

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'resnet_v1_50',
        is_training=False)

    eval_image_size = 72

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    image = convert_to_opencv(image)

    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    with tf.compat.v1.Session() as sess:
        try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: image})
        except KeyError:
            print("Couldn't find classification output layer: " + output_layer + ".")
            print("Verify this a model exported from an Object Detection project.")
            exit(-1)


    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    print('Classified as: ' + labels[highest_probability_index])
    print()

    # Or you can print out all of the results mapping labels to probabilities.
    label_index = 0
    for p in predictions:
        truncated_probablity = np.float64(np.round(p,8))
        print (labels[label_index], truncated_probablity)
        label_index += 1




if __name__ == '__main__':
  tf.app.run()
