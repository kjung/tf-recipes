################################################################################
# Sequence labeling example in tensorflow - the difference here is that each
# timestep has a _vector_ of labels instead of a single label.
# This works, but doesn't do particularly well.  For the default model, starting
# with embeddings initialized via word2vec skipgram embeddings, we start out
# with a validation set loss of 3.21, and after one epoch get down to 2.36.
# But subsequent epochs of training seem to start overfitting, and the
# validation loss creeps up again, ending up at 2.49.  But I don't care all
# that much, so I'm going to let it go...
# With hidden_dim = 200, we start at 2.91, and end up with a min at 2.34 again
# after 1 epoch.
# With hidden_dim = 400, does a bit worse...
# With d400 embeddings, and hidden_dim = 200, start out at 2.77 and get to 2.32
# at 1 epoch.
# With d400 embeddings, and hidden_dim = 400, start at 2.63 and get to 2.36.
# 
# Some other notes - the doctorAI paper and code uses 0/1 encodings of the targets
# and uses a cross entropy equation for the loss that doesn't really apply (as
# cross entropy though I suppose it's still a likelihood) as the label distribution
# isn't a distribution.  You can't use such a target with the standard tf cross
# entropy loss functions b/c it messes up the gradient calculations.  So to see
# what, if any difference that makes, tried both versions of the loss, monitoring
# loss, top-5 ppv and recall, with otherwise default parameters.  
# 
# Ran for max 15 epochs.  
# General - using original doctorAI loss, training is MUCH slower.
# Loss:
#   normalized target - started at 5.33, low of 2.41 after 2 epochs.
#   doctorAI - started at 7.34, low of 6.97 at 15 epochs.  
# PPV:
#   normalized target - started at 0.00321, best 0.138 after 2 epochs
#   doctorAI - started at 0.00507, best 0.982 at 15 epochs
# Recall:
#   normalized target - started at 0.0168, best 0.714 at 2 epochs
#   doctorAI - started at 0.0161, best of 0.409 at 15 epochs
# 
# Not much indication of overfitting to the training set since gap to validation
# was pretty small.
# The other interesting thing is that normalized targets led to very quick 
# saturation in performance, and subsequent decreases even in the training set,
# which I haven't seen very often before.  I'd have expected to see overfitting
# to the training set, so this is perhaps worth investigating.
# Note - the statistics files are in:
# 
# logs/statistics-doctorai-[vanilla|normalized-target].pkl
#
# TODO - Why not use tf.nn.sigmoid_cross_entropy_with_logits instead of all this
# silliness?!
# 
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
    X_train = pickle.load(open(os.path.join(FLAGS.data_dir, "doctorai_X_train.pkl"), "rb"))
    Y_train = pickle.load(open(os.path.join(FLAGS.data_dir, "doctorai_Y_train.pkl"), "rb"))    
    X_val = pickle.load(open(os.path.join(FLAGS.data_dir, "doctorai_X_val.pkl"), "rb"))
    Y_val = pickle.load(open(os.path.join(FLAGS.data_dir, "doctorai_Y_val.pkl"), "rb"))    
    train_iterator = PaddedSequenceVectorTargetIterator(X_train, Y_train, FLAGS.batch_size,
                                                        FLAGS.max_length, FLAGS.input_dim, FLAGS.output_dim)
    val_iterator = PaddedSequenceVectorTargetIterator(X_val, Y_val, FLAGS.batch_size,
                                                      FLAGS.max_length, FLAGS.input_dim, FLAGS.output_dim)
    return train_iterator, val_iterator

def build_graph() :
    input_dim = FLAGS.input_dim
    max_length = FLAGS.max_length
    embedding_dim = FLAGS.embedding_dim
    hidden_dim = FLAGS.hidden_dim
    output_dim = FLAGS.output_dim
    batch_size = FLAGS.batch_size

    tf.reset_default_graph()
    # Input placeholders
    # Note that if we want to do bucketing, we can leave the max_steps dimension unspecified.  
    input_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length, input_dim])
    target_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length, output_dim])
    seqlen_placeholder = tf.placeholder(tf.int32, shape=[batch_size,])
    loss_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length])

    # Embedding layer
    embeddings = tf.get_variable("embeddings", 
                                 [input_dim, embedding_dim],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    reshaped_input = tf.reshape(input_placeholder, [-1, input_dim])
    input_embeddings = tf.matmul(reshaped_input, embeddings)
    input_embeddings = tf.reshape(input_embeddings, [batch_size, max_length, embedding_dim])

    # Recurrent net
    rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
    initial_state = rnn_cell.zero_state(batch_size, tf.float32)
    rnn_output, rnn_final_state = tf.nn.dynamic_rnn(cell=rnn_cell, 
                                                    dtype=tf.float32, 
                                                    sequence_length=seqlen_placeholder, 
                                                    initial_state=initial_state, 
                                                    inputs=input_embeddings)
                                                
    rnn_output = tf.nn.dropout(rnn_output, tf.constant(0.5))

    # Output layer parameters; this is just a softmax over the vocabulary.  
    output_weights = tf.get_variable('weights', 
                                     [hidden_dim, output_dim], 
                                     initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    output_biases = tf.get_variable('bias', 
                                    [output_dim], 
                                    initializer=tf.constant_initializer(0.0))
                                
    # Outputs
    collapsed_outputs = tf.reshape(rnn_output, [-1, hidden_dim])
    logits = tf.matmul(collapsed_outputs, output_weights) + output_biases
    output_probs = tf.nn.softmax(logits)                                
    top_k = tf.nn.top_k(output_probs, 5)

    # Loss and optimizer
    reshaped_targets = tf.reshape(target_placeholder, [-1, output_dim])
    reshaped_loss_mask = tf.reshape(loss_mask_placeholder, [-1])
    # My custom loss function
    #loss = mean_sequence_cross_entropy_from_probs(output_probs, reshaped_targets, reshaped_loss_mask)
    # Using tf library function, which doesn't compute gradients correctly for this use case
    # (expects target rows to be valid categorical distributions).
    # Using cross entropy with soft labels    
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits, reshaped_targets)
    #loss = loss * tf.reshape(loss_mask_placeholder, [-1])
    #loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask_placeholder)
    # Using a tensorflow function that seems appropriate
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, reshaped_targets)
    loss = tf.reduce_mean(loss, axis=1)
    loss = loss * tf.reshape(loss_mask_placeholder, [-1])
    loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask_placeholder)

    G = { 'input_placeholder': input_placeholder,
          'target_placeholder': target_placeholder,
          'seqlen_placeholder': seqlen_placeholder,
          'loss_mask_placeholder': loss_mask_placeholder,
          'logits': logits,
          'output_probs': output_probs,
          'embeddings': embeddings, 
          'loss': loss,
          'top_k' : top_k }

    return G


def calc_ppv(targets, top_k, mask, k) :
    retval = 0.
    for i in range(len(targets)) :
        if mask[i] > 0 : 
            retval += np.sum(targets[i, top_k[i]]) / k
    return retval / np.sum(mask)

def calc_recall(targets, top_k, mask, k) :
    retval = 0.
    for i in range(len(targets)) :
        if mask[i] > 0 : 
            retval += np.sum(targets[i, top_k[i]])  / np.sum(targets[i,:])
    return retval / np.sum(mask)
    
    
def eval_on_dataset(sess, G, iterator, dataset_name="validation") :
    """Run evaluation sweep through iterator and print out results, resetting
       the epoch counter of the iterator to the original value afterwards. """
    print(">>> Evaluating model on %s" % (dataset_name))
    step = 0
    current_epoch = iterator.epoch
                
    # Evaluate against validation before training to get baseline performance!  
    step = 0
    cumulative_loss = 0.
    cumulative_ppv = 0.
    cumulative_recall = 0.
    while current_epoch == iterator.epoch : 
        step += 1
        this_x, this_y, this_seqlen, this_mask = iterator.next()
        feed_dict = {G['input_placeholder']: this_x, 
                     G['target_placeholder']: this_y, 
                     G['seqlen_placeholder']: this_seqlen, 
                     G['loss_mask_placeholder']: this_mask}
        loss_value, top_k_ = sess.run([G['loss'], G['top_k']], feed_dict=feed_dict)
        cumulative_loss += loss_value
        reshaped_y = this_y.reshape((FLAGS.batch_size*FLAGS.max_length, FLAGS.output_dim))
        reshaped_mask = this_mask.reshape((FLAGS.batch_size*FLAGS.max_length))
        cumulative_ppv += calc_ppv(reshaped_y, top_k_.indices, reshaped_mask, 5)
        cumulative_recall += calc_recall(reshaped_y, top_k_.indices, reshaped_mask, 5)
        
    val_loss = cumulative_loss / float(step)
    val_ppv = cumulative_ppv / float(step)
    val_recall = cumulative_recall / float(step)
    print(">>> Metrics in %s after epoch %d <<< " % (dataset_name, current_epoch))
    print("\t>>> Loss = %.4f, Recall = %.3f, PPV = %.4f <<<" % (val_loss, val_recall, val_ppv))
    iterator.epoch = current_epoch
    return val_loss, val_ppv, val_recall

def run_training() :
    # Set up local variables for convenience.
    init_lr = FLAGS.learning_rate
    input_dim = FLAGS.input_dim
    output_dim = FLAGS.output_dim
    max_length = FLAGS.max_length
    embedding_dim = FLAGS.embedding_dim
    hidden_dim = FLAGS.hidden_dim
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
    embeddings = G['embeddings']
    if FLAGS.embedding_file == "" :
        embeddings_init_values = None
    else : 
        print("Loading embedding initial values from %s" % (FLAGS.embedding_file))
        embeddings_init_values = np.array(pickle.load(open(os.path.join(FLAGS.data_dir, FLAGS.embedding_file), "rb")))

    # Add summary statistic node, set up optimizer and training op.
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(init_lr)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    # Some more setup... 
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    # Start session and init variables.
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    if not embeddings_init_values == None :
        sess.run(embeddings.assign(embeddings_init_values))

    # Some lists to accumulate statistics
    train_losses = []
    val_losses = []
    train_ppvs = []
    val_ppvs = []
    train_recalls = []
    val_recalls = []
        
    # Run through training and validation sets to get a baseline for how we do at the start.
    train_loss, train_ppv, train_recall = eval_on_dataset(sess, G, train_iterator, "Training Set")
    val_loss, val_ppv, val_recall = eval_on_dataset(sess, G, val_iterator, "Validation Set")
    train_losses.append(train_loss)
    train_ppvs.append(train_ppv)
    train_recalls.append(train_recall)
    val_losses.append(val_loss)
    val_ppvs.append(val_ppv)
    val_recalls.append(val_recall)
    
    # The training loop
    print("Starting training loop")
    step = 0
    current_epoch = 0
    while current_epoch < max_epochs : 
        start_time = time.time()
    
        # Train and accumulate statistics on one batch
        this_x, this_y, this_seqlen, this_mask = train_iterator.next()
        feed_dict = {input_placeholder: this_x, 
                     target_placeholder: this_y, 
                     seqlen_placeholder: this_seqlen, 
                     loss_mask_placeholder: this_mask}
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        step += 1
        duration = time.time() - start_time

        # Periodically save learning metrics.  
        if step % 200 == 0 : 
            print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if train_iterator.epoch > current_epoch : 
            current_epoch += 1
            step = 0            
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model_epoch_%d.ckpt' % (current_epoch))
            print("Saving checkpoint to %s" % (checkpoint_file))
            saver.save(sess, checkpoint_file, global_step=step)
            train_loss, train_ppv, train_recall = eval_on_dataset(sess, G, train_iterator, "Training Set")
            val_loss, val_ppv, val_recall = eval_on_dataset(sess, G, val_iterator, "Validation Set")
            train_losses.append(train_loss)
            train_ppvs.append(train_ppv)
            train_recalls.append(train_recall)
            val_losses.append(val_loss)
            val_ppvs.append(val_ppv)
            val_recalls.append(val_recall)
            val_iterator.epoch += 1
    statistics = {'train_loss' : train_losses,
                  'train_ppv' : train_ppvs,
                  'train_recall' : train_recall,
                  'val_loss' : val_losses,
                  'val_ppv' : val_ppvs,
                  'val_recall' : val_recalls}
    pickle.dump(statistics, open(os.path.join(FLAGS.log_dir, "statistics.pkl"), "wb"))

def main() :
    run_training()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--input_dim',
                        type=int,
                        default=15851,
                        help='input dimension')
    parser.add_argument('--output_dim',
                        type=int,
                        default=255,
                        help='output dimension')
    parser.add_argument('--max_length',
                        type=int,
                        default=10,
                        help="Max sequence length (longer sequences are truncated)")
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help="Dimension of word embeddings")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="Dimension of hidden state of RNN")
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help="Batch size")
    parser.add_argument('--embedding_file',
                        type=str,
                        default="",
                        help="Filename of embedding file for initializing embedding layer")
    parser.add_argument('--max_epochs',
                        type=int,
                        default=5,
                        help="Max number of epochs to train")
    parser.add_argument('--data_dir',
                        type=str,
                        default="/data1/stride6/data",
                        help="Directory where data files reside")
    parser.add_argument('--log_dir',
                        type=str,
                        default="/home/kjung/projects/tf-recipes/logs",
                        help="Directory to put log files and model checkpoints")

    # Note - FLAGS is global...  
    FLAGS, unparsed_args = parser.parse_known_args()
    run_training()
    
