import numpy as np
import skimage.data
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer

np.random.seed(42)

# load data
camera = skimage.data.camera()
astronaut = rgb2gray(skimage.data.astronaut())
horse = skimage.data.horse()
coffee = rgb2gray(skimage.data.coffee())

# merge data
data_list = [camera, astronaut, horse, coffee]

def preprocess_image(img, w=128, h=128):
    # resize image
    img = resize(img, (w, h), mode='reflect')

    # thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2 * (binary * 1) - 1 # bool to int

    # reshape
    return np.reshape(shift, w*h)

data_list = [preprocess_image(d) for d in data_list]

# Create Amari Hopfield Network Model (discrete mode by default)
model = AmariHopfieldNetwork(num_neurons=data_list[0].shape[0], asyn=False, activation="sign")
model.init_state()
trainer = HebbianTrainer(model)
trainer.train(data_list)

# Generate testset
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

tests = [get_corrupted_input(d, 0.3) for d in data_list]

# Use the new predict_batch method for better progress reporting
predicted = trainer.predict_batch(tests)

# display predict results
def plot(data, test, predicted, figsize=(5, 6)):
    def reshape(data):
        dim = int(np.sqrt(len(data)))
        data = np.reshape(data, (dim, dim))
        return data

    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("discrete_hopfield_train.png")
    plt.show()


plot(data_list, tests, predicted, figsize=(5, 6))
