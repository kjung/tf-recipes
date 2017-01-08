################################################################################
# Sequence labeling example in tensorflow - language model using data from a
# twitter dataset.  Model outputs a distribution over vocab at each step.  
################################################################################
# These should appear in all... 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import math
from datetime import datetime
from six.moves import xrange

import cPickle as pickle
import numpy as np
import tensorflow as tf

# Import from util.py...  
from util import *

def load_data() :
    # Load data from file (pickled lists of lists)
    print("Reading data...")
    X = pickle.load(open(os.path.join(FLAGS.data_dir, "X_train.pkl"), "rb"))
    Y = pickle.load(open(os.path.join(FLAGS.data_dir, "Y_train.pkl"), "rb"))
    index_to_word = pickle.load(open(os.path.join(FLAGS.data_dir, "index_to_word.pkl"), "rb"))
    word_to_index = pickle.load(open(os.path.join(FLAGS.data_dir, "word_to_index.pkl"), "rb"))

    print("Splitting data into train and validation...")
    indices = np.random.permutation(np.arange(len(Y)))
    stop_index = np.floor(len(indices) * 0.8).astype(np.int64)
    train_indices = indices[:stop_index]
    val_indices = indices[stop_index:]
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_val = X[val_indices]
    Y_val = Y[val_indices]

    train_iterator = PaddedSequenceIterator(X_train, Y_train, FLAGS.batch_size, FLAGS.max_length)
    val_iterator = PaddedSequenceIterator(X_val, Y_val, FLAGS.batch_size, FLAGS.max_length)

    return train_iterator, val_iterator

def build_graph() :
    vocab_size = FLAGS.vocab_size
    max_length = FLAGS.max_length
    embedding_dim = FLAGS.embedding_dim
    hidden_size = FLAGS.hidden_size
    output_dim = vocab_size
    batch_size = FLAGS.batch_size

    tf.reset_default_graph()
    # Input placeholders
    # Note that if we want to do bucketing, we can leave the max_steps dimension unspecified.  
    input_placeholder = tf.placeholder(tf.int32, shape=[batch_size, max_length])
    target_placeholder = tf.placeholder(tf.int32, shape=[batch_size, max_length])
    seqlen_placeholder = tf.placeholder(tf.int32, shape=[batch_size,])
    loss_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length])

    # Embedding layer
    embeddings = tf.get_variable("embeddings", 
                                 [vocab_size, embedding_dim],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    input_embeddings = tf.nn.embedding_lookup(embeddings, input_placeholder)

    # Recurrent net
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=0.0)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    rnn_output, rnn_final_state = tf.nn.dynamic_rnn(cell=lstm_cell, 
                                                    dtype=tf.float32, 
                                                    sequence_length=seqlen_placeholder, 
                                                    initial_state=initial_state, 
                                                    inputs=input_embeddings)
                                                
    rnn_output = tf.nn.dropout(rnn_output, tf.constant(0.5))

    # Output layer parameters; this is just a softmax over the vocabulary.  
    output_weights = tf.get_variable('weights', 
                                     [hidden_size, vocab_size], 
                                     initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    output_biases = tf.get_variable('bias', 
                                    [vocab_size], 
                                    initializer=tf.constant_initializer(0.0))
                                
    # Outputs
    collapsed_outputs = tf.reshape(rnn_output, [-1, hidden_size])
    logits = tf.matmul(collapsed_outputs, output_weights) + output_biases
    output_probs = tf.nn.softmax(logits)                                

    # Loss and optimizer
    one_hot_target = tf.one_hot(tf.reshape(target_placeholder, [-1]), depth=vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_target)
    loss = loss * tf.reshape(loss_mask_placeholder, [-1])
    loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask_placeholder)

    G = { 'input_placeholder': input_placeholder,
          'target_placeholder': target_placeholder,
          'seqlen_placeholder': seqlen_placeholder,
          'loss_mask_placeholder': loss_mask_placeholder,
          'logits': logits,
          'output_probs': output_probs,
          'loss': loss}

    return G

def run_training() :
    # Set up local variables for convenience.
    init_lr = FLAGS.learning_rate
    vocab_size = FLAGS.vocab_size
    max_length = FLAGS.max_length
    embedding_dim = FLAGS.embedding_dim
    hidden_size = FLAGS.hidden_size
    output_dim = vocab_size
    batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs    

    # Load datasets and create dataset iterators.
    train_iterator, val_iterator = load_data()

    # Build computation graph; build_graph returns a dictionary of
    # graph elements that we'll need.  
    G = build_graph()
    
    # Set up some local references...
    input_placeholder = G['input_placeholder']
    target_placeholder = G['target_placeholder']
    seqlen_placeholder = G['seqlen_placeholder']
    loss_mask_placeholder = G['loss_mask_placeholder']
    loss = G['loss']
    output_probs = G['output_probs']

    # Add summary statistic node, set up optimizer and training op.
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(init_lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Add code to calculate number correct, and accuracy... 
    # To calculate the number correct, we want to count padded steps as incorrect
    predicted_classes = tf.cast(tf.argmax(output_probs,1), tf.int32)
    num_correct = tf.cast(tf.equal(predicted_classes, tf.reshape(target_placeholder, [-1])), tf.int32)
    num_correct *= tf.cast(tf.reshape(loss_mask_placeholder, [-1]), tf.int32)
    accuracy = tf.reduce_sum(tf.cast(num_correct, tf.float32) / tf.cast(tf.reduce_sum(loss_mask_placeholder), tf.float32))

    # Some more setup... 
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()

    # Start session and init variables.
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # The training loop
    print("Starting training loop")
    step, acc = 0, 0 
    tr_acc, val_acc = [], []
    current_epoch = 0
    while current_epoch < max_epochs : 
        start_time = time.time()
    
        # Train and accumulate statistics on one batch
        this_x, this_y, this_seqlen, this_mask = train_iterator.next()
        feed_dict = {input_placeholder: this_x, 
                     target_placeholder: this_y, 
                     seqlen_placeholder: this_seqlen, 
                     loss_mask_placeholder: this_mask}
        _, loss_value, acc_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
        acc += acc_value
        step += 1
        duration = time.time() - start_time

        # Periodically save learning metrics.  
        if step % 100 == 0 : 
            print('Step %d: loss = %.2f, acc = %.2f (%.3f sec)' % (step, loss_value, acc_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if train_iterator.epoch > current_epoch : 
            current_epoch += 1
            tr_acc.append(acc / step)

            # Save checkpoint model
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model_epoch_%d.ckpt' % (current_epoch))
            print("Saving checkpoint to %s" % (checkpoint_file))
            saver.save(sess, checkpoint_file, global_step=step)
                
            # Evaluate against validation
            step, acc = 0, 0
            val_epoch = val_iterator.epoch
            while val_epoch == val_iterator.epoch : 
                step += 1
                this_x, this_y, this_seqlen, this_mask = val_iterator.next()
                feed_dict = {input_placeholder: this_x, 
                             target_placeholder: this_y, 
                             seqlen_placeholder: this_seqlen, 
                             loss_mask_placeholder: this_mask}
                _, loss_value, acc_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
                acc += acc_value
            val_acc.append(acc / step)
            step, acc = 0, 0
            print("Accuracy after epoch", current_epoch, " - tr:", tr_acc[-1], "- te:", val_acc[-1])


def main() :
    run_training()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--vocab_size',
                        type=int,
                        default=8000,
                        help='Vocabulary size')
    parser.add_argument('--max_length',
                        type=int,
                        default=10,
                        help="Max sequence length (longer sequences are truncated)")
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help="Dimension of word embeddings")
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help="Dimension of hidden state of RNN")
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help="Batch size")
    parser.add_argument('--max_epochs',
                        type=int,
                        default=5,
                        help="Max number of epochs to train")
    parser.add_argument('--data_dir',
                        type=str,
                        default="/home/kjung/projects/tf-recipes/data",
                        help="Directory where data files reside")
    parser.add_argument('--log_dir',
                        type=str,
                        default="/home/kjung/projects/tf-recipes/logs",
                        help="Directory to put log files and model checkpoints")

    # Note - FLAGS is global...  
    FLAGS, unparsed_args = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed_args)
    run_training()
    
