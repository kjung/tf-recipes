import numpy as np
import tensorflow as tf

# Simple iterator for variable length sequences.  Handles padding of inputs
# and targets, and also returns sequence lengths and a loss mask.
class PaddedSequenceIterator : 
    def __init__(self, inputs, targets, batch_size, max_length) : 
        self.inputs = inputs
        self.targets = targets
        self.N = len(inputs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.epoch = 0
        self.indices = np.random.permutation(np.arange(self.N))
        self.cursor = 0
    
    def next(self) :
        batch_size = self.batch_size
        max_length = self.max_length
        if self.cursor + batch_size - 1 > self.N : 
            self.epoch += 1
            self.indices = np.random.permutation(np.arange(self.N))
            self.cursor = 0
        input_batch = self.inputs[self.indices[self.cursor:self.cursor+self.batch_size]]
        target_batch = self.targets[self.indices[self.cursor:self.cursor+self.batch_size]]
        self.cursor += batch_size
        
        # Deal with variable sequence lengths by padding; create mask along the way. 
        # Note - for sequences longer than max_length, we take the last max_length elements.  
        seqlen = np.asarray([len(x) if len(x) < max_length else max_length for x in input_batch], dtype=np.int32)
        inputs = np.zeros([batch_size, max_length], dtype=np.int32)
        targets = np.zeros([batch_size, max_length], dtype=np.int32)
        mask = np.zeros([batch_size, max_length], dtype=np.float32)
        for i, (x_i, y_i) in enumerate(zip(inputs, targets)) : 
            if len(input_batch[i]) >= self.max_length : 
                x_i = input_batch[i][-max_length:]
                y_i = target_batch[i][-max_length:]
            else :
                x_i[:len(input_batch[i])] = input_batch[i]
                y_i[:len(target_batch[i])] = target_batch[i]
            mask[i][:seqlen[i]] = 1
        
        return inputs, targets, seqlen, mask


class PaddedSequenceVectorTargetIterator : 
    def __init__(self, inputs, targets, batch_size, max_length, input_dim, output_dim) : 
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.N = len(inputs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epoch = 0
        self.indices = np.random.permutation(np.arange(self.N))
        self.cursor = 0

    def next(self) : 
        batch_size = self.batch_size
        max_length = self.max_length
        input_dim = self.input_dim
        output_dim = self.output_dim
        if self.cursor + batch_size - 1 > self.N : 
            self.epoch += 1
            self.indices = np.random.permutation(np.arange(self.N))
            self.cursor = 0
        input_batch = self.inputs[self.indices[self.cursor:self.cursor+self.batch_size]]
        target_batch = self.targets[self.indices[self.cursor:self.cursor+self.batch_size]]
        self.cursor += batch_size
        
        # Deal with variable sequence lengths by padding; create mask along the way. 
        # Note - for sequences longer than max_length, we take the last max_length elements.  
        inputs = np.zeros([batch_size, max_length, input_dim], dtype=np.float32)
        targets = np.zeros([batch_size, max_length, output_dim], dtype=np.float32)
        seqlen = np.asarray([len(x) if len(x) < max_length else max_length for x in input_batch], dtype=np.int32)
        mask = np.zeros([batch_size, max_length], dtype=np.float32)

        # i indexes through sequences in the batch
        for i, (x_i, y_i) in enumerate(zip(inputs, targets)) : 
            if len(input_batch[i]) >= self.max_length : 
                input_i = input_batch[i][-max_length:]
                target_i = target_batch[i][-max_length:]
            else : 
                input_i = input_batch[i]
                target_i = target_batch[i]
            # j indexes through elements of the sequence
            for j in range(len(input_i)) :
                x_i[j, input_i[j] ] = 1
                # Note - we don't have to guard for dividing by 0 because we assume a no-output-code code,
                # so len(target_i[j]) is at least 1.
                # The first (commented) line is in case we are using the vanila tensorflow
                # cross entropy loss function, which assumes the targets are valid distributions
                # when computing the gradient.  The Theano code for doctorAI on the other hand,
                # just uses a bag of codes vector as a target, and still applies the cross
                # entropy loss.  So testing that out to see how it behaves.  
                #y_i[j, target_i[j] ] = 1. / len(target_i[j])
                y_i[j, target_i[j] ] = 1. 
            mask[i][:seqlen[i]] = 1.

        return inputs, targets, seqlen, mask


class ReverseSequenceLabelIterator : 
    def __init__(self, inputs, targets, batch_size, max_length, input_dim) : 
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.N = len(inputs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.input_dim = input_dim
        self.epoch = 0
        self.indices = np.random.permutation(np.arange(self.N))
        self.cursor = 0

    def next(self) : 
        batch_size = self.batch_size
        max_length = self.max_length
        input_dim = self.input_dim
        if self.cursor + batch_size - 1 > self.N : 
            self.epoch += 1
            self.indices = np.random.permutation(np.arange(self.N))
            self.cursor = 0
        input_batch = self.inputs[self.indices[self.cursor:self.cursor+self.batch_size]]
        target_batch = self.targets[self.indices[self.cursor:self.cursor+self.batch_size]]
        self.cursor += batch_size
        
        # Deal with variable sequence lengths by padding; create mask along the way. 
        # Note - for sequences longer than max_length, we take the last max_length elements.  
        inputs = np.zeros([batch_size, max_length, input_dim], dtype=np.float32)
        targets = np.zeros([batch_size], dtype=np.float32)
        seqlen = np.asarray([len(x) if len(x) < max_length else max_length for x in input_batch], dtype=np.int32)
        mask = np.zeros([batch_size, max_length], dtype=np.float32)

        # i indexes through sequences in the batch
        for i, (x_i, y_i) in enumerate(zip(inputs, targets)) : 
            if len(input_batch[i]) >= self.max_length : 
                input_i = input_batch[i][-max_length:][::-1]
            else : 
                input_i = input_batch[i][::-1]
            targets[i] = target_batch[i]
            for j in range(len(input_i)) :
                x_i[j, input_i[j] ] = 1
            mask[i][:seqlen[i]] = 1.

        return inputs, targets, seqlen, mask


class SequenceLabelIterator : 
    def __init__(self, inputs, targets, batch_size, max_length, input_dim) : 
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.N = len(inputs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.input_dim = input_dim
        self.epoch = 0
        self.indices = np.random.permutation(np.arange(self.N))
        self.cursor = 0

    def next(self) : 
        batch_size = self.batch_size
        max_length = self.max_length
        input_dim = self.input_dim
        if self.cursor + batch_size - 1 > self.N : 
            self.epoch += 1
            self.indices = np.random.permutation(np.arange(self.N))
            self.cursor = 0
        input_batch = self.inputs[self.indices[self.cursor:self.cursor+self.batch_size]]
        target_batch = self.targets[self.indices[self.cursor:self.cursor+self.batch_size]]
        self.cursor += batch_size
        
        # Deal with variable sequence lengths by padding; create mask along the way. 
        # Note - for sequences longer than max_length, we take the last max_length elements.  
        inputs = np.zeros([batch_size, max_length, input_dim], dtype=np.float32)
        targets = np.zeros([batch_size], dtype=np.float32)
        seqlen = np.asarray([len(x) if len(x) < max_length else max_length for x in input_batch], dtype=np.int32)
        mask = np.zeros([batch_size, max_length], dtype=np.float32)

        # i indexes through sequences in the batch
        for i, (x_i, y_i) in enumerate(zip(inputs, targets)) : 
            if len(input_batch[i]) >= self.max_length : 
                input_i = input_batch[i][-max_length:]
            else : 
                input_i = input_batch[i]
            targets[i] = target_batch[i]
            for j in range(len(input_i)) :
                x_i[j, input_i[j] ] = 1
            mask[i][:seqlen[i]] = 1.

        return inputs, targets, seqlen, mask
        


def mean_sequence_cross_entropy_from_probs(probs, targets, loss_mask) : 
    """Return mean cross entropy loss for the input sequences whose logits are in outputs
       and with target values in targets.  Note dimensions for the outputs and targets!  
    Args: 
      logits - tensor of class probs of shape [batch_size * max_length, num_classes].  
      targets - tensor of class labels of shape [batch_size * max_length, num_classes]. 
      loss_mask - tensor of 0/1 values used for masking the loss, of shape [batch_size, max_length]
    Returns: 
      1-D tensor of size batch_size with cross entropy of each sequence in the batch.  
    """

    logEPS = 1e-8
    probs = tf.nn.softmax(probs)
    
    # Calculate cross entropy for each row of logits.  Now have a [batch_size x max_length, num_classes] tensor
    # Use the Theano code from doctorAI.py as a template for this...
    cross_entropy = -1. * (targets * tf.log(probs + logEPS) - (1. - targets) * tf.log(1. - probs + logEPS))
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy *= tf.reshape(loss_mask, [-1])
    
    # Sum across time steps, including the zeros.  
    sum_cross_entropy = tf.reduce_sum(cross_entropy)
    
    # Divide the sum cross entropy by number of unmasked timesteps.
    mean_cross_entropy = sum_cross_entropy / tf.reduce_sum(loss_mask)
    return mean_cross_entropy
