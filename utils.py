
import os
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_curve(step,train_loss, val_loss, filename):

    fig = plt.figure(figsize=(12,8))
    plt.plot(step, train_loss, color='r', clip_on=False, label='Training Loss')
    plt.plot(step, val_loss, color='b', clip_on=False, label='Validation Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    fig.savefig(filename, dpi = 600, bbox_inches = 'tight')
    plt.close()


def plot_training_curve(tensorboard_log, output_directory):

    df = pd.read_csv(tensorboard_log, header=None, names=['Iteration', 'Training_Loss', 'Validation_Loss'])
    step = df['Iteration'].to_numpy()  # Changed from as_matrix() to to_numpy()
    train_loss = df['Training_Loss'].to_numpy()
    val_loss = df['Validation_Loss'].to_numpy()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plot_curve(step=step, train_loss=train_loss, val_loss=val_loss, 
               filename=os.path.join(output_directory, 'training_validation_loss.png'))

if __name__ == '__main__':
    
    plot_training_curve(tensorboard_log = 'log/train_log.csv', output_directory = 'figures')
