import argparse
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from PIL import Image
from torchvision import transforms, datasets
from face_reco import preprocessing
from face_reco.face_features_extractor import FaceFeaturesExtractor
from face_reco.face_recogn import FaceRecogn

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model.')
    parser.add_argument('-d', '--dataset-path',
                        help='Path to folder with images.')

    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    # Transforms are common image transformations.
    # They can be chained together using Compose.
    # Composes several transforms together
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []

    # dataset.samples ---> (img_path, label) ---> ('C:\\Users\\Korisnik\\Desktop\\Face-Recognition\\images\\Bon Jovi\\Bon Jovi (6).jpg', 0)

    for img_path, label in dataset.samples:
        print(img_path)

        # returns all bbs and embendings of single image
        _, embedding = features_extractor(
            transform(Image.open(img_path).convert('RGB')))

        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue

        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def load_data(args, features_extractor):
    # A generic data loader
    dataset = datasets.ImageFolder(args.dataset_path)

    # returns embedding of every image
    embeddings, labels = dataset_to_embeddings(
        dataset, features_extractor)

    return embeddings, labels, dataset.class_to_idx


def train(embeddings, labels):
    # Logistic Regression classifier
    # uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’
    softmax = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    softmax.fit(embeddings, labels)

    return softmax


def main():
    # Parser for command-line options, arguments and sub-commands
    args = parse_args()

    # Object of class FaceFeaturesExtractor
    features_extractor = FaceFeaturesExtractor()

    # returns embeddings of all faces in all pictures
    # returns labels of all pitcures
    # class_to_idx ---> 'Bon Jovi' : 0, 'Brad Pit' : 1 .....
    embeddings, labels, class_to_idx = load_data(
        args, features_extractor)

    # returns fitted estimator
    clf = train(embeddings, labels)

    # idx_to_class ---> 0: 'Bon Jovi' ....
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(
        idx_to_class.items(), key=lambda i: i[0]))

    print(metrics.classification_report(labels, clf.predict(
        embeddings), target_names=list(target_names)))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)

    result = FaceRecogn(clf, idx_to_class)
    res = result()

    clf_path = os.path.join('model', 'clf.pkl')

    pkl1 = open(clf_path, "wb")
    pickle.dump(res, pkl1)

    pkl1.close()


main()
