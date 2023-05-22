from eeg_calculate_features import calculate_features
from LSTM_unsupervised import train_lstm
from confusion_matrix_generator import plot_confusion_matrix

'''
Main file for calculating features and training the LSTM.
'''
if __name__ == '__main__':
    # Run eeg_calculate_features for each participant number.
    calculate_features(1)
    calculate_features(2)
    calculate_features(3)
    calculate_features(5)
    calculate_features(6)

    # Start training on model/LSTM_unsupervised.py. 
    # Inputs: train1, train2, train3, train4, test1, epochs, no_of_features, no_of_channels.
    tn, fp, fn, tp = train_lstm(1, 2, 3, 6, 5, 10, 4, 5)

    # Plot confusion matrix
    plot_confusion_matrix(tn, fp, fn, tp)
