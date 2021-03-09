# Autocomplete / Language Generation Model
* High-level Tasks
 * Corpus Pre-Processing
 * Building the Language Model
 * Language Model Generalization

* Corpus Pre-Processing
 * Use text from WoS
 * Create a toy data set, and a full data set
 * Use articles from on specific WoS area for 2020, say chemistry
 * Sample down to ~100 records for toy data set
 * Add n-1 start tokens "<s>"
 * Add a single end token "</s>"

* Build language model
 * Count matrix - row is occurances on (i-2, i-1) bigram, col is count of word following that bigram
 * Probability matrix - divide each cell by row sum
 * Language model - a script that uses the probablity matrix, estimate the probability of a given sentence  
 * Log probability

* Language Model Generalization
 * Perplexity score for evaluation
 * Out of vocab words
 * Smoothing

* Notes
 * Treat each sentence in the corpus as an "observation"


## Classes
### Autocomplete (what would sklearn do?)
* Methods
 * Fit
 * Predict next word (autocomplete)
 * Generate text (predict next word until end of sentence toke. do K times, specified in method call)
 * Save model
 * Load model
* Attributes
 * Count matrix
 * Propbability matrix

