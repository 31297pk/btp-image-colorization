from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import array

PATHNAME = "./datasets/pokemon/"
TRAIN_DIR_NAME = "train/"
TEST_DIR_NAME = "test/"

NUM_TRAIN_IMAGES = 819
NUM_TEST_IMAGES = 819

BATCH_SIZE = 20
NUM_LABELS = 1 # Set to 1 if no labels present

# This is for pokemon dataset only -----------------------
# This is for pokemon dataset only
LABELS_DATASET = []
gen_size = [820]

for i in range(len(gen_size)):
    for j in range(gen_size[i]):
        LABELS_DATASET.append(i)

# One hot encoding

LABELS_DATASET = array(LABELS_DATASET).reshape([-1, 1])
ohe = OneHotEncoder(sparse=False)
LABELS_DATASET = ohe.fit_transform(LABELS_DATASET)
LABEL_NAMES = [
                'pokemon',
              ]

