import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history):
    '''
    Takes history of our model and plot accuracy and loss function

    parms:
    history - model history

    output:
    -
    '''
    loss = history.history["loss"]
    acc = history.history["accuracy"]
    number_of_elements = len(loss)
    epochs = np.linspace(1,number_of_elements,number_of_elements)

    ## Plotting functions
    fig,ax = plt.subplots(2,1, figsize=(10,7))
    ax[0].plot(epochs, acc)
    ax[0].set_title("Accuracy")  
    ax[1].plot(epochs, loss)
    ax[1].set_title("Loss function")
    plt.show()

def show_confusion_matrix(model,X,y_true):
    pred = model.predict(X)
    confusion_matrix = confusion_matrix(y_true, pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()