Download Link: https://assignmentchef.com/product/solved-cs114-assignment3-n-gram-language-models
<br>
The overall goal of this assignment is for you to train the best language model possible given the same limited amount of data. The data folder should contain the training set (train-data.txt), the development set (dev-data.txt), and a lot of jumbled data (jumble-dev). Test sets (for sentences and jumbled data) have been held out and are not given to you.

You are also given train-data small.txt and test-data small.txt, the toy corpus from HW2, in the same format. You should only create more models similar to the one in unigram.py, while leaving the other files intact.

<h1>Assignment</h1>

<h2>Unigram model</h2>

The unigram model, unigram.py, is given to you; you do not need to (and should not) change anything. However, before creating any new models, you should read the code and understand what it is doing. In particular, note the following:

train(self, trainingSentences) – Trains the model from the supplied collection of trainingSentences. First, we count all the words in the training data. Note that one symbol LanguageModel.STOP is counted at the end of each sentence, and the unknown word LanguageModel.UNK is also given a count of 1.

Then we build self.prob counter as a scipy.sparse.lil matrix. For the unigram model, this is not necessary, but for bigram models with large vocabularies, it is impossible to store all of the probabilities in a Numpy array without running out of memory. However, using a sparse matrix (here, in list of lists format), we can take advantage of the fact that most of the bigram probabilities are 0, and store only the non-zero probabilities, saving a lot of space. There are a few idiosyncrasies (e.g. division becomes multiplication by the reciprocal), but for the most part, we can use Scipy sparse matrices similarly to Numpy arrays.

As in PA2, self.word dict should be used to translate between indices and words; by popular demand, this time we can use a

{word:index} dictionary rather than {index:word}. Note the use of self.total to store the denominator—in this case, the total number of words in the training data. Finally, we divide by self.total to get the probabilities <em>P</em>(word) in self.prob counter.

getWordProbability(self, sentence, index) – Returns the probability of the word at index, according to the model, within the specified sentence. Note that if the index is at the end of the sentence, we return the probability of LanguageModel.STOP, and if we have not seen the word at index before, we return the probability of LanguageModel.UNK.

generateWord(self, context) – Returns, for a given context, a random word, according to the probabilities in the model. The context is a list of the previous words in the sentence. For the unigram model, the context is ignored, so we simply use numpy.random.choice to generate a word using the unigram probabilities in self.prob counter.

<h2>Bigram model</h2>

Now that you are familiar with how the unigram model works, your first task is to create a bigram model in bigram.py. Whereas a unigram model uses probabilities of words <em>P</em>(word), a bigram model uses conditional probabilities <em>P</em>(word|previous word). Therefore, instead of just storing a single probability as self.probCounter[word], we can set self.prob counter[previous word][word] = <em>P</em>(word|previous word). Similarly, self.total[previous word] stores the count of previous word in the training data.

Some notes:

For a word at the beginning of the sentence (i.e. the index is 0 in getWordProbability, or the context is empty in generateWord), the previous “word” will be LanguageModel.START.

When building a conditional probability table (as in HW2), there is no row for &lt;/S&gt;, and no column for &lt;S&gt;. Probably the easiest way to account for this in your code is to use a single index to represent

LanguageModel.START when used for a row and LanguageModel.STOP when used for a column.

If you get a divide-by-zero warning when calculating the (dev) test set/jumbled sentence perplexity, that is good! Do not worry about that.

<h2>Bigram model with add-<em>k </em>smoothing</h2>

Your next task is to create a bigram model with add-<em>k </em>smoothing, in bigram add k.py. A na¨ıve implementation would involve calculating the ad-

justed probabilities <em>P</em><sup>ˆ</sup>(<em>w<sub>n</sub></em>|<em>w<sub>n</sub></em><sub>−1</sub>) = count(<em>w<sup>n</sup></em><sup>−1</sup><em>,w<sup>n</sup></em>) + <em>k </em>and storing them count(<em>w<sub>n</sub></em><sub>−1</sub>) + <em>k</em>|<em>V </em>|

in self.prob counter. Do not do this! (You will run out of memory.) Instead, we will use some tricks:

<ol>

 <li>Collect the bigram counts as before, such that prob counter only contains those bigrams that have non-zero actual counts in the training data.</li>

 <li>Add <em>k</em>|<em>V </em>| to total[previous word], so that after dividing, the count(<em>w<sub>n</sub></em>−<sub>1</sub><em>,w<sub>n</sub></em>)</li>

</ol>

values of self.prob counter have form       .

count(<em>w<sub>n</sub></em><sub>−1</sub>) + <em>k</em>|<em>V </em>|

<ol start="3">

 <li>Then, whenever we need a probability (array), we can add the missing <em>k </em>inside of getWordProbability and</li>

</ol>

count(<em>w<sub>n</sub></em><sub>−1</sub>) + <em>k</em>|<em>V </em>|

generateWord.

Otherwise, your procedure should be the same as for the unsmoothed bigram model.

<strong>Setting the value of </strong><em>k</em>

Which value of <em>k </em>should you use? When you are first writing your code, you can use <em>k </em>= 1 (i.e., add-one smoothing) for simplicity. After you get your algorithm working, though, you should try to find the value of <em>k </em>that results in the best performance on the dev set, either in terms of minimum perplexity, or maximum accuracy on the jumble task, or both.

<h2>Bigram model with interpolation</h2>

Finally, you should create a bigram model with (simple linear) interpolation, in bigram interpolation.py. If you implemented bigram.py correctly, you should be able to leave train and getWordProbability (and init ) as is, only needing to fill in generateWord.

<strong>Tuning the interpolation weights</strong>

As above, you should experiment with finding the values of <em>λ </em>that optimize the dev set perplexity and/or jumble task accuracy. Both books mention the EM (expectation-maximization) algorithm, but for now, trial and error is fine.

<h1>Evaluation</h1>

You are given a shell script, run, that automatically launches the evaluator and runs some tests. If you edit this file, you can change the language model that is used for evaluation by editing the model parameter, e.g.

–model unigram.Unigram uses the Unigram model provided. tester.py tests the following things:

<ol>

 <li>Makes sure the probability sums to one given a random context.</li>

 <li>Computes the perplexity on the training and (dev) test sets. Edit the test parameter in the run script to change which data set is used for testing.</li>

 <li>The jumbled sentence task. This will use your LM to find which sentence is a real sentence out of 10 jumbled sentences. All of these tests are given to you. This should make your evaluation very consistent and easy to compare across models.</li>

</ol>

<h1>Write-up</h1>

Write a short report on the language models that you have explored. You should at least describe the following (which will count toward your grade):

How you set the value of <em>k</em>/tuned the interpolation weights

Perplexity on the training set and dev set for each model

Performance of each model on the jumbled sentence task

Note that you only need to report on models trained/tested on the full data (you do not need to include any results on the toy data).

Please also include the following (which will <em>not </em>count toward your grade):

(About) how many hours you spent working on this assignment

Any parts of the assignment you found particularly easy or difficult

Any other comments on the assignment you would like to make