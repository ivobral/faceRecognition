import argparse
import pickle
import os
from face_reco.face_features_extractor import FaceFeaturesExtractor
from face_reco import preprocessing
from PIL import Image
from .util import draw_bb_on_img


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for detecting and classifying faces on user-provided image. This script will process image, draw '
        'bounding boxes and labels on image and display it. It will also optionally save that image.')
    parser.add_argument('--image-path', required=True,
                        help='Path to image file.')
    parser.add_argument(
        '--save-dir', help='If save dir is provided image will be saved to specified directory.')

    return parser.parse_args()


def recognise_faces(img, feature_extractor):

    bbs, embedding = feature_extractor(img)

    if embedding is None:
        print("Could not find face!!!")

    return embedding, bbs


def classification(embedding, bbs, img):
    pkl = open('model/clf.pkl', "rb")
    clf = pickle.load(pkl)
    i = 0

    prediction = clf[0].classifier.predict_proba(embedding)

    for bb in bbs:
        draw_bb_on_img(bb, img, prediction[i], clf[0].idx_to_class)
        i = i + 1

    return img


def main():
    args = parse_args()

    feature_extractor = FaceFeaturesExtractor()
    preprocess = preprocessing.ExifOrientationNormalize()
    img = Image.open(args.image_path)
    # == args.image_path
    filename = img.filename
    img = preprocess(img)
    img = img.convert('RGB')

    # returns embeddings of all detected faces in image
    # returns bbs of all detected faces in image
    embedding, bbs = recognise_faces(img, feature_extractor)

    # returns boundboxed image with name and prediction in %
    result_img = classification(embedding, bbs, img)

    if args.save_dir:
        basename = os.path.basename(filename)
        name = basename.split('.')[0]
        ext = basename.split('.')[1]
        result_img.save('{}/{}_tested_image.{}'.format(
            args.save_dir, name, ext))

    result_img.show()


main()
