from collections import Counter
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np


# f4nctiNone, learning_rate='optimal'o load in movie review data
def split_data(file):
    file = open(file, 'r')
    reviews = file.readlines()
    stop_index = len(reviews) * .8
    index = 0
    training = []
    test = []
    while index < stop_index:
        training.append(reviews[index])
        index += 1
    while index < len(reviews) - 1:
        test.append(reviews[index])
        index += 1
    return training, test


# function to fit data to model. Defined prior to
# creating vocab in global scope
def custom_fit(data):
    unique_words = set()

    for each_sentence in data:
        for each_word in each_sentence.split(' '):
            # print(each_word)

            if len(each_word) > 2:
                unique_words.add(each_word)

    vocab = {}

    for index, word in enumerate(sorted(list(unique_words))):
        vocab[word] = index

    # print(vocab)
    return vocab


# movie review data
training_pos, test_pos = split_data('rt-polarity.pos')
training_neg, test_neg = split_data('rt-polarity.neg')

# list containing combined positive and negative reviews for training
training_full = []
training_full.extend(training_pos)
training_full.extend(training_neg)
# list containing combined positive and negative reviews for testing
test_full = []
test_full.extend(test_pos)
test_full.extend(test_neg)

# global vocab used in custom_transform()
vocab = custom_fit(training_full)


# function to transform data after it has been fitted to the model
def custom_transform(data):

    # vocab = custom_fit(data)

    row, col, val = [], [], []

    for idx, sentence in enumerate(data):

        count_word = dict(Counter(sentence.split(' ')))
        for word, count in count_word.items():

            if len(word) > 2:
                col_index = vocab.get(word)
                # test for None type first to catch runtime errors
                if col_index is not None and col_index >= 0:
                    row.append(idx)
                    col.append(col_index)
                    val.append(count)

    return csr_matrix((val, (row, col)), shape=(len(data), len(vocab)))


# targets for training data
targets_training = np.ones(len(training_full))
i = len(training_pos)
while i < len(training_full):
    targets_training[i] = 2
    i += 1

# targets for test data
targets_test = np.ones(len(test_full))
i = len(test_pos)
while i < len(test_full):
    targets_test[i] = 2
    i += 1

target_names = ['positive', 'negative']

tfidf_transformer_1 = TfidfTransformer()
custom_transform_training_counts = custom_transform(training_full)
custom_training_tfidf_counts = tfidf_transformer_1.fit_transform(custom_transform_training_counts)
classifier = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-4, random_state=42,
                           max_iter=5, tol=None, learning_rate='optimal').fit(custom_training_tfidf_counts, targets_training)
predicted = classifier.predict(custom_training_tfidf_counts)
print('mean correct classification for custom_transform with training data: ',
      np.mean(predicted == targets_training))
print(metrics.classification_report(targets_training, predicted,
                                    target_names=target_names))
custom_transform_test_counts = custom_transform(test_full)
custom_test_tfidf_counts = tfidf_transformer_1.fit_transform(custom_transform_test_counts)
predicted = classifier.predict(custom_test_tfidf_counts)
print('\nmean correct classification for custom_transform with test data: ', np.mean(predicted == targets_test))
print(metrics.classification_report(targets_test, predicted,
                                    target_names=target_names))

cv = CountVectorizer()
tfidf_transformer_2 = TfidfTransformer()
CV_training_counts = cv.fit_transform(training_full)
CV_training_tfidf_counts = tfidf_transformer_2.fit_transform(CV_training_counts)
classifier = SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-4, random_state=42,
                           max_iter=5, tol=None, learning_rate='optimal').fit(CV_training_tfidf_counts, targets_training)
predicted = classifier.predict(CV_training_tfidf_counts)
print('\nmean correct classification for CountVectorizer with training data: ', np.mean(predicted == targets_training))
print(metrics.classification_report(targets_training, predicted,
                                    target_names=target_names))
CV_test_counts = cv.transform(test_full)
CV_test_tfidf_counts = tfidf_transformer_2.transform(CV_test_counts)
predicted = classifier.predict(CV_test_tfidf_counts)
print('\nmean correct classification for CountVectorizer with test data: ', np.mean(predicted == targets_test))
print(metrics.classification_report(targets_test, predicted,
                                    target_names=target_names))

