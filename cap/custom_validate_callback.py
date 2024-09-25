import tensorflow as tf
import numpy as np
import csv

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name):
        super(CustomCallback, self).__init__()
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # evaluate the model every n epochs
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            # model.evaluate_generator ---> model.evaluate
            loss, acc = self.model.evaluate(self.test_generator, steps=self.test_steps, verbose=0)
                                
            print(f'\nValidation loss: {loss}, acc: {acc}\n')
            
            self.write_val_to_csv(epoch, loss, acc)

    # validation metrics to csv file
    def write_val_to_csv(self, epoch, loss, acc):
        with open(self.model_name + 'validation_metrics.csv', 'a', newline='') as csvFile:
            metricWriter = csv.writer(csvFile)
            metricWriter.writerow([epoch, loss, acc])