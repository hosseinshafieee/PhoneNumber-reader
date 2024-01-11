from keras.models import load_model
import cv2
import numpy as np
from mltu.utils.text_utils import ctc_decoder
import tensorflow as tf

class CTCloss(tf.keras.losses.Loss):
    """ CTCLoss objec for training the model"""
    def __init__(self, name: str = "CTCloss", reduction='') -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Compute the training batch CTC loss value"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss
    
    
class CWERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).
    
    Args:
        padding_token: An integer representing the padding token in the input data.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, padding_token=4, name="CWER", **kwargs):
        # Initialize the base Metric class
        super(CWERMetric, self).__init__(name=name, **kwargs)
        
        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the padding token as an attribute
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Convert the dense decode tensor to a sparse tensor
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)
        
        # Convert the dense true labels tensor to a sparse tensor and cast to int64
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        # Retain only the non-padding elements in the predicted labels tensor
        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))
        
        # Retain only the non-padding elements in the true labels tensor
        true_labels_sparse = tf.sparse.retain(true_labels_sparse, tf.not_equal(true_labels_sparse.values, self.padding_token))

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        
        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(len(y_true))

        # Calculate the number of wrong words in batch and add to wer_accumulator variable
        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32)))

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A dictionary containing the CER and WER.
        """
        return {
                "CER": tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32)),
                "WER": tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
        }

# crop boxes in image 
def crop_boxes(img, bounding_box):
    return img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]


#  load the trained model
model = load_model('./Models/phone_reader.h5', custom_objects={'CTCloss': CTCloss, 'CWERMetric': CWERMetric})





numbers = []
import os
from tqdm import tqdm
j = 0
for i in tqdm(os.listdir('./Datasets/number')):
    images = []
    j += 1
    try:
        img = cv2.imread(f'./Datasets/number/{i}', cv2.IMREAD_GRAYSCALE)

        img = crop_boxes(img, (int(200), int(1200), int(900), int(1320)))
        median = np.array(img).mean()

        if (median > 127):
            ret, img = cv2.threshold(img, median, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, img = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 10, 10, 50, 50, cv2.BORDER_CONSTANT,value=(0, 0, 0))
        img = cv2.resize(img, (128, 32))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        images.append(np.array(img))

    except Exception as e:
        print('first error', e)
                

    try:
        img = cv2.imread(f'./Datasets/number/{i}', cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (200, 50))

        img = crop_boxes(img, (int(200), int(1230), int(900), int(1320)))

        median = np.array(img).mean()

        if (median > 127):
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
        img = cv2.copyMakeBorder(img, 15, 15, 80, 80, cv2.BORDER_CONSTANT,value=(0, 0, 0))
        img = cv2.resize(img, (128, 32))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        images.append(img)
    except Exception as e:
            print('second error', e)

    try:
        img = cv2.imread(f'./Datasets/number/{i}', cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (200, 50))

        img = crop_boxes(img, (int(120), int(600), int(950), int(750)))

        median = np.array(img).mean()

        if (median > 127):
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 15, 15, 80, 80, cv2.BORDER_CONSTANT,value=(0, 0, 0))
        img = cv2.resize(img, (128, 32))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        images.append(img)
    except Exception as e:
        print('third error', e)

    try: 
        img = cv2.imread(f'./Datasets/number/{i}', cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (200, 50))

        img = crop_boxes(img, (int(150), int(200), int(950), int(380)))

        median = np.array(img).mean()

        if (median > 127):
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 15, 15, 80, 80, cv2.BORDER_CONSTANT,value=(0, 0, 0))
        img = cv2.resize(img, (128, 32))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        images.append(img)
    except Exception as e:
        print('fourth error', e)
        
        
    result1 = model.predict(np.array(images).reshape((-1,  32, 128, 1)), verbose=0)
            
    text = ctc_decoder(result1, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    if (len(text) > 0 and len(text[0]) == 11):
        numbers.append(text[0])
    if (len(text) > 1 and len(text[1]) == 11):
        numbers.append(text[1])
    if (len(text) > 2 and len(text[2]) == 11):
        numbers.append(text[2])
    if (len(text) > 3 and len(text[3]) == 11):
        numbers.append(text[3])


    
with open(f'./numbers.txt', 'w') as f:
    f.write('\n'.join(numbers))

# remove repeated phone numbers
words = open('./numbers.txt', "r").readlines()
numbers = []
for k in range(1, 6893):
    label = words[k].replace('\n', '')
    # print (label)
    numbers.append(label)
numbers = list(set(numbers))
print(len(numbers))


with open(f'./numbers2.txt', 'w') as f:
    f.write('\n'.join(numbers))