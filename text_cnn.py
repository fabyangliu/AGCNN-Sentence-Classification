import tensorflow as tf
import math


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, 
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        def print_shape(t):
            print(t.op.name,' ',t.get_shape().as_list())

        def activate(t,b,label):
            if label=='nlrelu':
                return tf.log(tf.nn.relu(t+b)+1.)
            elif label=='selu':
                return tf.nn.selu(t+b)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #self.W_1 = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev= 0.05), trainable = True)
            self.W_1 = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev= math.sqrt(1.0/(vocab_size*embedding_size))), trainable = True)

            self.embedded_chars_1 = tf.nn.embedding_lookup(self.W_1, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars_1, -1)#[batch_size,sequence_length,embedding_size,1]

           
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        filter_size_2_list = [1,3,5]
        num_filters_2 = 2
        for filter_size_ in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size_):
                # Convolution Layer
                filter_shape = [filter_size_, embedding_size, 1, num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                #W1 = tf.get_variable("w1",shape=filter_shape,initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")#[none,sequence_length-filter_size+1,1,num_filters]
                # Apply nonlinearity
                #conv = tf.contrib.layers.batch_norm(conv)
                h = activate(conv,b1,'nlrelu')
                h = tf.reshape(h,[-1,sequence_length - filter_size_ + 1,num_filters,1])
                print_shape(h)
                
                for filter_size_2 in filter_size_2_list:
                    W2 = tf.Variable(tf.truncated_normal([filter_size_2, 1, 1, num_filters_2], stddev=0.1), name="W2")
                    b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b1")
                    #W3 = tf.get_variable("W3",shape=[3, 1, 1, 1],initializer=tf.contrib.layers.xavier_initializer())
                    conv = tf.nn.conv2d(
                        h,
                        W2,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")#[none,sequence_length-filter_size_+1,num_filters,1]
                    # Apply nonlinearity
                    #conv = tf.contrib.layers.batch_norm(conv)
                    conv = h * activate(conv,b2,'nlrelu')
                    print_shape(conv)
                    
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        conv,
                        ksize=[1, sequence_length - filter_size_+1, 1, 1],#
                        strides=[1, 1, 1, 1],#liang ge fangxiang buchang wei 1
                        padding='VALID',
                        name="pool2")
                    print_shape(pooled)
                    pooled_outputs.append(pooled)#[none,1,num_filters,100]

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) *len(filter_size_2_list) * num_filters_2
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W4 = tf.get_variable(
                "W4",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #W4 = tf.Variable(tf.truncated_normal([int(num_filters_total/2), num_classes], stddev=math.sqrt(2.0/(num_classes*int(num_filters_total/2)))),name='w4')
            b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W4, b4, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) 

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
