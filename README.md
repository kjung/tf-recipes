Tensorflow Recipes
=========================================

Some examples of recent deep neural net models used mostly on healthcare data, re-implemented in Tensorflow.  
The models were chosen as examples of various architectural choices, and include a language model, doctorAI, 
and RETAIN.  

The language model is an example of sequence labeling (i.e., an output at each timestep, as we go), with 
one hot inputs and one hot targets.  

DoctorAI is an example of sequence labeling, with outputs at each timestep as we go, but with bag of words
inputs and outputs.  I tried various loss functions, including the original, plus some others that make more
sense to me. 

RETAIN is an example of a simple attention model, with a single binary target at the end of each sequence, 
with the recurrent parts used to generate weights on each sequence element, and each dimension of the sequence
elements.  

There are a few Jupyter notebooks that were used to work through the model architectures interactively, prepare 
datasets, etc.  

Note that no data is included for obvious reasons.   

** Requirements **
Mostly tested and run using Tensorflow rc 0.12.  

