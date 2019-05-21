import tensorflow as tf
import numpy as np
import pickle as pk
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

# Parameters
# ==================================================
activation_label = 'nlrelu'
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/CR-data/positive", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/CR-data/negative", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 25, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps")
# Misc Parameters
#tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

cnn = TextCNN(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(vocab_processor.vocabulary_),
    embedding_size=FLAGS.embedding_dim,
    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
    num_filters=FLAGS.num_filters,
    l2_reg_lambda=FLAGS.l2_reg_lambda)

# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
lr_in = tf.placeholder(tf.float32, name="input_lr")
#lr_new = tf.train.exponential_decay(lr_in,global_step,2000,0.95,True)
optimizer = tf.train.AdamOptimizer(lr_in)
grads = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads, global_step=global_step)
init = tf.global_variables_initializer()
        
def train_step(x_batch, y_batch, lr):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
      lr_in: lr
    }
    _, step, loss, accuracy = sess.run(
        [train_op, global_step, cnn.loss, cnn.accuracy],
        feed_dict)
    #time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

def dev_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    step, loss, accuracy = sess.run(
        [global_step, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return time_str, step, loss, accuracy
acc_list = []
acc_accumulate = 0.0
for i in range(1):
    with tf.Session(config = config) as sess:
        # Initialize all variables
        sess.run(init)
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Early stopping
        max_acc = 0.0
        current_step = 0
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, 1e-3)
            lr_label = 1
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("Evaluation:")
                time_str, step, loss, acc = dev_step(x_dev, y_dev)
                if acc > max_acc:
                    max_acc = acc
                print("{}: step {}, loss {:g}, acc {:g}, max_acc {:g}, lr_label {}".format(time_str, step, loss, acc, max_acc, lr_label))
    acc_accumulate += max_acc
    acc_list.append(max_acc)
    
for i in range(8):
    #cross_validation
    dev_index_r = -1 * int(0.1 * float(i+2) * float(len(y_shuffled)))
    dev_index_l = -1 * int(0.1 * float(i+1) * float(len(y_shuffled)))
    x_train, x_dev = np.concatenate((x_shuffled[:dev_index_r],x_shuffled[dev_index_l:]),axis = 0), x_shuffled[dev_index_r : dev_index_l]
    y_train, y_dev = np.concatenate((y_shuffled[:dev_index_r],y_shuffled[dev_index_l:]),axis = 0), y_shuffled[dev_index_r : dev_index_l]
    
    for j in range(1):
        with tf.Session(config = config) as sess:
            # Initialize all variables
            sess.run(init)
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Early stopping
            max_acc = 0.0
            current_step = 0
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, 1e-3)
                lr_label = 1
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation:")
                    time_str, step, loss, acc = dev_step(x_dev, y_dev)
                    if acc > max_acc:
                        max_acc = acc
                    print("{}: step {}, loss {:g}, acc {:g}, max_acc {:g}, lr_label {}".format(time_str, step, loss, acc, max_acc, lr_label))
        acc_accumulate += max_acc
        acc_list.append(max_acc)
#cross_validation
dev_sample_index = -1 * int(0.9 * float(len(y_shuffled)))
x_train, x_dev = x_shuffled[dev_sample_index:], x_shuffled[:dev_sample_index]
y_train, y_dev = y_shuffled[dev_sample_index:], y_shuffled[:dev_sample_index]
del x_shuffled, y_shuffled
for j in range(1):
    with tf.Session(config = config) as sess:
        # Initialize all variables
        sess.run(init)
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Early stopping
        max_acc = 0.0
        current_step = 0
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, 1e-3)
            lr_label = 1
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("Evaluation:")
                time_str, step, loss, acc = dev_step(x_dev, y_dev)
                if acc > max_acc:
                    max_acc = acc
                print("{}: step {}, loss {:g}, acc {:g}, max_acc {:g}, lr_label {}".format(time_str, step, loss, acc, max_acc, lr_label))
    acc_accumulate += max_acc
    acc_list.append(max_acc)
acc = acc_accumulate /10.0

f=open('./data/accuracy_record','a+',encoding='utf-8')
time_string=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
f.write('Date:'+time_string+'\t Average Accuracy: %g'%acc+'\t Activation Label: %s \n'%activation_label)
f.close()
