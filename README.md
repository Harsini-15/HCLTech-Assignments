Multiclass Sentiment Analysis
Neural Network, RNN, Bidirectional LSTM and Bidirectional GRU with Automated Hyperparameter Optimization
Dataset: Sp1786/multiclass-sentiment-analysis-dataset
Classes: Negative, Neutral, Positive
Platform: Google Colab (GPU strongly recommended)
Framework: TensorFlow / Keras

This notebook investigates whether improved text representation, bidirectional sequence learning and systematic hyperparameter optimization improve multiclass sentiment classification. Four comparable architectures are implemented: a feedforward Neural Network, Simple RNN, Bidirectional LSTM and Bidirectional GRU.

1.Loaded the multiclass sentiment dataset (Negative, Neutral, Positive) from Hugging Face with predefined train, validation and test splits.
2.Cleaned the data, normalized URLs and user mentions and tokenized the text.
3.Initialized the embedding layer with pretrained GloVe vectors and handled class imbalance with class weights.
4.Implemented four models: Neural Network (NN), Simple RNN, Bidirectional LSTM and Bidirectional GRU.
5.Tuned hidden units, dropout, learning rate and batch size automatically using random search over a predefined grid (12 trials per model), with early stopping and learning-rate reduction.
6.Selected each model's best configuration by validation macro-F1, retrained it and evaluated it once on the test set using accuracy, precision, recall and macro-F1.
7.Identified the best-performing model and its hyperparameters (printed by the cell below).
8.Results come from a single run with one random seed, so models with very close scores should be treated as comparable rather than definitively ranked.
