import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix_per(tn, fp, fn, tp):
    # Create confusion matrix
    cm = np.array([[tn, fp], 
                [fn, tp]])

    # Convert to percentages
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)

    # Create labels
    classes = ['0', '1']

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Configure the plot
    disp = disp.plot(cmap=plt.cm.Blues)


    # Increase font size
    for text in disp.text_:
        for t in text:
            t.set_fontsize(15)

    plt.show()

def plot_confusion_matrix(tn, fp, fn, tp):
    # Create confusion matrix
    cm = np.array([[tn, fp], 
                [fn, tp]])

    # Create labels
    classes = ['0', '1']

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Configure the plot
    disp = disp.plot(cmap=plt.cm.Blues)


    # Increase font size
    for text in disp.text_:
        for t in text:
            t.set_fontsize(15)

    plt.show()

#plot_confusion_matrix(8776, 4193, 32, 18)
