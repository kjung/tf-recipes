################################################################################
# Re-implementation of Baha's RETAIN model, which uses a simple embedding for
# time series of medical codes, along with a two-way attention mechanism via
# GRU based recurrent nets.  The writeup is rather muddled, so rather than
# speculate about wtf they are doing, I will refer to their implementation,
# written in Theano.
# One note - they support dropout after the embedding layer though there is
# no mention of it in the paper.  They also do dropout before prediction.
# The default parameters for their program have it _on_, which is curious... 
# Also - doesn't look like there explicit loss masking done in their code, 
# despite variable length sequences?
# Generally, with a target of - is there a ccs code after the last timestep,
# (not a meaningful target, just for convenience), the model with default
# settings quickly overfits, and reaches max AUPRC after 2 epochs.  After
# that, AUROC keeps increasing, but AUPRC drops...  Then at some point,
# everything starts to diverge and we get stuck with totally random outputs
# with AUROC = 0.500.  Should figure out what is going on here.  
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# Import from util.py...  
from util import *

def load_data() :
    # Load data from file (pickled lists of lists)
    print("Reading data...")
    X_train = pickle.load(open(os.path.join(FLAGS.data_dir, "retain_X_train.pkl"), "rb"))
    Y_train = pickle.load(open(os.path.join(FLAGS.data_dir, "retain_Y_train.pkl"), "rb"))    
    X_val = pickle.load(open(os.path.join(FLAGS.data_dir, "retain_X_val.pkl"), "rb"))
    Y_val = pickle.load(open(os.path.join(FLAGS.data_dir, "retain_Y_val.pkl"), "rb"))    
    train_iterator = ReverseSequenceLabelIterator(X_train, Y_train, FLAGS.batch_size,
                                                  FLAGS.max_length, FLAGS.input_dim)
    val_iterator = ReverseSequenceLabelIterator(X_val, Y_val, FLAGS.batch_size,
                                         FLAGS.max_length, FLAGS.input_dim)
    return train_iterator, val_iterator

def build_graph() :
    input_dim = FLAGS.input_dim
    output_dim = 2
    batch_size = FLAGS.batch_size    
    max_length = FLAGS.max_length
    embedding_dim = FLAGS.embedding_dim
    hidden_dim = FLAGS.hidden_dim
    l2_coeff = FLAGS.l2_coeff
    init_lr = FLAGS.init_lr

    tf.reset_default_graph()
    # Input placeholders
    # Note that if we want to do bucketing, we can leave the max_steps dimension unspecified.  
    input_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length, input_dim])
    target_placeholder = tf.placeholder(tf.float32, shape=[batch_size, ])
    seqlen_placeholder = tf.placeholder(tf.int32, shape=[batch_size,])
    loss_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_length])

    # Embedding layer
    embeddings = tf.get_variable("embeddings", 
                                 [input_dim, embedding_dim],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    reshaped_input = tf.reshape(input_placeholder, [-1, input_dim])
    input_embeddings = tf.matmul(reshaped_input, embeddings)
    input_embeddings = tf.reshape(input_embeddings, [batch_size, max_length, embedding_dim])
    input_embeddings = tf.nn.dropout(input_embeddings, tf.constant(0.5))

    # Recurrent net alpha (scalar weights on visit representations)
    with tf.variable_scope('rnn_alpha') :
        rnn_cell_alpha = tf.nn.rnn_cell.GRUCell(hidden_dim)
        alpha_initial_state = rnn_cell_alpha.zero_state(batch_size, tf.float32)
        rnn_alpha_output, rnn_alpha_final_state = tf.nn.dynamic_rnn(cell=rnn_cell_alpha,
                                                                    dtype=tf.float32,
                                                                    sequence_length=seqlen_placeholder,
                                                                    initial_state=alpha_initial_state,
                                                                    inputs=input_embeddings)
    W_alpha = tf.get_variable('W_alpha',
                              [hidden_dim,1],
                              initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    bias_alpha = tf.get_variable('bias_alpha', [1,])

    alpha_scores = tf.matmul(tf.reshape(rnn_alpha_output, [-1, hidden_dim]), W_alpha) + bias_alpha
    alpha_scores = tf.reshape(alpha_scores, [batch_size, max_length])
    alpha_scores = alpha_scores * loss_mask_placeholder + EPS
    alpha = tf.nn.softmax(alpha_scores)

    # Recurrent net for generating per dimension weights
    with tf.variable_scope('rnn_beta') :
        rnn_cell_beta = tf.nn.rnn_cell.GRUCell(hidden_dim)
        beta_initial_state = rnn_cell_beta.zero_state(batch_size, tf.float32)
        rnn_beta_output, rnn_beta_final_state = tf.nn.dynamic_rnn(cell=rnn_cell_beta,
                                                                  dtype=tf.float32,
                                                                  sequence_length=seqlen_placeholder,
                                                                  initial_state=beta_initial_state,
                                                                  inputs=input_embeddings)
    W_beta = tf.get_variable('W_beta',
                             [hidden_dim, embedding_dim],
                             initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
    bias_beta = tf.get_variable('bias_beta', [embedding_dim,],
                                initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    beta = tf.tanh(tf.matmul(tf.reshape(rnn_beta_output, [-1, hidden_dim]), W_beta) + bias_beta)
    beta = tf.reshape(beta, [batch_size * max_length, embedding_dim])
    
    # Calculate context vectors
    context = beta * tf.reshape(input_embeddings, [-1, embedding_dim])
    context = tf.nn.dropout(context, tf.constant(0.5))
    context = tf.reshape(context, [batch_size * max_length, embedding_dim])

    # Classification layer
    W_classify = tf.get_variable('W_classify', [embedding_dim,1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
    bias_classify = tf.get_variable('bias_classify', [1])
    logits = tf.matmul(context, W_classify) + bias_classify
    logits = tf.reshape(logits, [batch_size, max_length])
    logits = tf.gather_nd(logits, tf.pack([tf.range(batch_size), seqlen_placeholder - 1], axis=1))
    output_probs = tf.nn.sigmoid(logits)

    cross_entropy_loss = -1. * target_placeholder * tf.log(output_probs) + (1. - target_placeholder)*(1. - output_probs)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    l2_loss = FLAGS.l2_coeff * tf.nn.l2_loss(W_classify)
    total_loss = cross_entropy_loss + l2_loss
    
    G = { 'input_placeholder': input_placeholder,
          'target_placeholder': target_placeholder,
          'seqlen_placeholder': seqlen_placeholder,
          'loss_mask_placeholder': loss_mask_placeholder,
          'logits': logits,
          'output_probs': output_probs,
          'embeddings': embeddings, 
          'loss': total_loss }


    return G

def eval_on_dataset(sess, G, iterator, dataset_name="validation") :
    """Run evaluation sweep through iterator and print out results, resetting
       the epoch counter of the iterator to the original value afterwards. """
    print(">>> Evaluating model on %s" % (dataset_name))
    step = 0
    current_epoch = iterator.epoch
                
    # Evaluate against validation before training to get baseline performance!  
    step = 0
    cumulative_loss = 0.0
    all_probs = np.array([], dtype=np.float32)
    all_targets = np.array([], dtype=np.float32)
    while current_epoch == iterator.epoch : 
        step += 1
        this_x, this_y, this_seqlen, this_mask = iterator.next()
        feed_dict = {G['input_placeholder']: this_x, 
                     G['target_placeholder']: this_y, 
                     G['seqlen_placeholder']: this_seqlen, 
                     G['loss_mask_placeholder']: this_mask}
        loss_value, probs = sess.run([G['loss'], G['output_probs']], feed_dict=feed_dict)
        cumulative_loss += loss_value
        all_probs = np.append(all_probs, probs)
        all_targets = np.append(all_targets, this_y)
    val_loss = cumulative_loss / float(step)
    auroc = roc_auc_score(all_targets, all_probs)
    auprc = average_precision_score(all_targets, all_probs)
    print(">>> (%s) After epoch %d, loss =  %.4f, auroc = %.4f, auprc = %.4f " % (dataset_name, current_epoch, val_loss, auroc, auprc))
    iterator.epoch = current_epoch

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

    # Run through training and validation sets to get a baseline for how we do at the start.
    eval_on_dataset(sess, G, train_iterator, "Training Set")
    eval_on_dataset(sess, G, val_iterator, "Validation Set")
        
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
            print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
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
            eval_on_dataset(sess, G, train_iterator, "Training Set")
            eval_on_dataset(sess, G, val_iterator, "Validation Set")
            val_iterator.epoch += 1
            

def main() :
    run_training()

if __name__ == '__main__' :
    EPS = 1e-9
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
    parser.add_argument('--l2_coeff',
                        type=float,
                        default=0.0001,
                        help='l2 penalty coefficient')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=5,
                        help="Max number of epochs to train")
    parser.add_argument('--init_lr',
                        type=float,
                        default=0.0001,
                        help='Initial learning rate')
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
    
