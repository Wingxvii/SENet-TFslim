import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

with tf.Session() as sess:

    # Let's read our pbtxt file into a Graph protobuf
    f = open('C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/Scene_labeler/data/graph.pbtxt', "r")
    graph_def_proto = text_format.Parse(f.read(), tf.GraphDef())

    # Import the graph protobuf into our new graph.
    graph_clone = tf.Graph()
    with graph_clone.as_default():
        g_in = tf.import_graph_def(graph_def=graph_def_proto)

    # Display the graph inline.
    graph_clone.as_graph_def()

LOGDIR='C:/Users/turnt/OneDrive/Desktop/Rob0Workspace/Scene_labeler'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

