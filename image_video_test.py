#!/usr/bin/env python3

import os
import cv2
import glob2

# import insightface
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from face_detector import YoloV5FaceDetector

BASE_URL = "https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/"
BUILDIN_RECOGNITION_MODELS = {
    "EfficientNetV2B0": {
        "file": "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_basic_agedb_30_epoch_17_0.977333.h5",
        "file_hash": "288f77378c0f5edd9acf102a8420165b",
    },
}


class FaceDetectorAndRecognizer:
    def __init__(self, recognize_model="EfficientNetV2B0", known_users=None, batch_size=32, force_reload=False):
        if recognize_model in BUILDIN_RECOGNITION_MODELS:
            file = BUILDIN_RECOGNITION_MODELS[recognize_model]["file"]
            file_hash = BUILDIN_RECOGNITION_MODELS[recognize_model]["file_hash"]
            recognize_model = os.path.join(os.path.expanduser("~"), ".keras", "models", file)
            if not os.path.exists(recognize_model):
                url = os.path.join(BASE_URL, file)
                print(">>>> Downloading from {} to {}".format(url, recognize_model))
                recognize_model = tf.keras.utils.get_file(file, url, cache_subdir="models", file_hash=file_hash)
        self.recognize_model = tf.keras.models.load_model(recognize_model, compile=False) if recognize_model is not None else None
        self.detect_model = YoloV5FaceDetector()
        if known_users is not None:
            assert self.recognize_model is not None, "recognize_model is not provided while initializing this instance."
            self.known_image_classes, self.known_embeddings = self.process_known_user_dataset(known_users, batch_size, force_reload=force_reload)
        else:
            self.known_image_classes, self.known_embeddings = None, None

    def image_detect_and_embedding(self, image, image_format="RGB"):
        bbs, _, ccs, nimgs = self.detect_model.detect_in_image(image, image_format=image_format)
        if len(bbs) == 0:
            return np.array([]), [], []
        emb_unk = self.recognize_model((nimgs - 127.5) * 0.0078125).numpy()
        emb_unk = emb_unk / np.linalg.norm(emb_unk, axis=-1, keepdims=True)
        return emb_unk, bbs, ccs

    def compare_images(self, images, image_format="RGB"):
        gathered_emb, gathered_bbs, gathered_ccs = [], [], []
        for id, image in enumerate(images):
            emb_unk, bbs, ccs = self.image_detect_and_embedding(image, image_format=image_format)
            gathered_emb.append(emb_unk)
            gathered_bbs.append(bbs)
            gathered_ccs.append(ccs)
            print(">>>> image_path: {}, faces count: {}".format(image if isinstance(image, str) else id, emb_unk.shape[0]))
        gathered_emb = np.concatenate(gathered_emb, axis=0)
        cosine_similarities = gathered_emb @ gathered_emb.T
        return cosine_similarities, gathered_emb, gathered_bbs, gathered_ccs

    def process_known_user_dataset(self, known_users, batch_size=32, force_reload=False):
        known_users = os.path.abspath(known_users)  # get rid of all ending "/" if any
        dest_pickle = os.path.join(known_users, os.path.basename(known_users) + "_embedding.npz")

        if force_reload == False and os.path.exists(dest_pickle):
            print(">>>> reloading known users from:", dest_pickle)
            aa = np.load(dest_pickle)
            image_classes, embeddings = aa["image_classes"], aa["embeddings"]
        else:
            if not os.path.exists(known_users):
                return [], []
            image_names = glob2.glob(os.path.join(known_users, "*/*.jpg"))

            """ Detct faces in images, keep only one face if exists in image. """
            nimgs, image_classes = [], []
            for image_name in tqdm(image_names, "Detect"):
                nimg = self.detect_model.detect_in_image(image_name, image_format="RGB")[-1]
                if nimg.shape[0] > 0:
                    nimgs.append(nimg[0])
                    image_classes.append(os.path.basename(os.path.dirname(image_name)))

            """ Extract embedding info from aligned face images """
            steps = int(np.ceil(len(image_classes) / batch_size))
            nimgs = (np.array(nimgs) - 127.5) * 0.0078125
            embeddings = [self.recognize_model(nimgs[ii * batch_size : (ii + 1) * batch_size]) for ii in tqdm(range(steps), "Embedding")]
            embeddings = np.concatenate(embeddings, axis=0)

            embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            image_classes = np.array(image_classes)
            np.savez_compressed(dest_pickle, embeddings=embeddings, image_classes=image_classes)
            print(">>>> saved known users to:", dest_pickle)

        print(">>>> image_classes info:")
        image_classes_counts = {}
        for ii in image_classes:
            image_classes_counts[ii] = image_classes_counts.get(ii, 0) + 1
        print("\n".join(["{}  {}".format(kk, image_classes_counts[kk]) for kk in sorted(image_classes_counts.keys())]))
        return image_classes, embeddings

    def search_in_known_users(self, image, image_format="RGB"):
        assert self.known_image_classes is not None and self.known_embeddings is not None, "known_users is not provided while initializing this instance."
        emb_unk, bbs, ccs = self.image_detect_and_embedding(image, image_format=image_format)
        cosine_similarities = emb_unk @ self.known_embeddings.T
        recognition_indexes = cosine_similarities.argmax(-1)
        recognition_similarities = [cosine_similarities[id, ii] for id, ii in enumerate(recognition_indexes)]
        recognition_classes = [self.known_image_classes[ii] for ii in recognition_indexes]
        return recognition_similarities, recognition_classes, bbs, ccs

    @staticmethod
    def draw_polyboxes_on_image(
        image, bboxes, detection_confidences=None, recognition_similarities=None, recognition_classes=None, similarity_thresh=0.6, draw_scale=1
    ):
        """bboxes: format `[[left, top, right, bottom]]`"""
        image = cv2.imread(image) if isinstance(image, str) else image
        for id, bbox in enumerate(bboxes):
            # Red color for unknown, green for Recognized
            similarity = 1 if recognition_similarities is None else recognition_similarities[id]
            color = (0, 0, 255) if similarity < similarity_thresh else (0, 255, 0)

            left, top, right, bottom = [int(ii) for ii in bbox]
            cv2.line(image, (left, top), (right, top), color, int(2 * draw_scale), cv2.LINE_AA)
            cv2.line(image, (right, top), (right, bottom), color, int(2 * draw_scale), cv2.LINE_AA)
            cv2.line(image, (right, bottom), (left, bottom), color, int(2 * draw_scale), cv2.LINE_AA)
            cv2.line(image, (left, bottom), (left, top), color, int(2 * draw_scale), cv2.LINE_AA)

            text_xx, text_yy, text_infos = np.max([left - 10, 10]), np.max([top - 10, 10]), []
            if recognition_classes is not None:
                label = "Unknown" if similarity < similarity_thresh else recognition_classes[id]
                text_infos.append("Label: {}, similarity: {:.4f}".format(label, similarity))
            if detection_confidences is not None:
                text_infos.append("det score: {}".format(detection_confidences[id]))
            cv2.putText(image, ", ".join(text_infos), (text_xx, text_yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * draw_scale, color, int(2 * draw_scale))
        return image

    def video_recognize(self, video_source=0, frames_per_detect=5, similarity_thresh=0.6, draw_scale=1):
        cap = cv2.VideoCapture(video_source)
        cur_frame_idx = 0
        while True:
            grabbed, frame = cap.read()
            if grabbed != True:
                break
            if cur_frame_idx % frames_per_detect == 0:
                if self.known_image_classes is not None:
                    recognition_similarities, recognition_classes, bbs, ccs = self.search_in_known_users(frame, image_format="BGR")
                else:  # Do detection only
                    bbs, _, ccs, nimgs = self.detect_model.detect_in_image(frame, image_format="BGR")
                    recognition_similarities, recognition_classes = None, None
                cur_frame_idx = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite("{}.jpg".format(cur_frame_idx), frame)
            if key == ord("q"):
                break

            self.draw_polyboxes_on_image(frame, bbs, ccs, recognition_similarities, recognition_classes, similarity_thresh, draw_scale=draw_scale)
            cv2.imshow("", frame)
            cur_frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import argparse

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--recognize_model", default="EfficientNetV2B0", help="Saved basic_model file path, NOT model")
    parser.add_argument("-k", "--known_users", type=str, default=None, help="Folder containing user images data, will do detetion only if not provided")
    parser.add_argument("-f", "--force_reload", action="store_true", help="Force reload known_user, not loading from saved cache")
    parser.add_argument("-b", "--embedding_batch_size", type=int, default=4, help="Batch size for extracting known user embedding data")
    parser.add_argument("-v", "--video_source", type=str, default="0", help="Video source")
    parser.add_argument(
        "-i", "--images", nargs="*", type=str, help="Image file pathes, will search in `known_users` if provided, or compare between provided images"
    )
    parser.add_argument("-t", "--similarity_thresh", type=float, default=0.6, help="Cosine similarity thresh, similarity lower than this will be Unknown")
    parser.add_argument("-p", "--frames_per_detect", type=int, default=5, help="Do detect every [NUM] frame")
    args = parser.parse_known_args(sys.argv[1:])[0]

    mm = FaceDetectorAndRecognizer(args.recognize_model, args.known_users, args.embedding_batch_size, force_reload=args.force_reload)
    if args.images is None or len(args.images) == 0:
        video_source = int(args.video_source) if str.isnumeric(args.video_source) else args.video_source
        mm.video_recognize(video_source, args.frames_per_detect, args.similarity_thresh)
    elif args.known_users is not None:
        for image in args.images:
            recognition_similarities, recognition_classes, bbs, ccs = mm.search_in_known_users(image)
            print("recognition_similarities:", recognition_similarities, "\nrecognition_classes:", recognition_classes, "\nbbs:", bbs, "\nccs:", ccs)

            dest_image = mm.draw_polyboxes_on_image(image, bbs, ccs, recognition_similarities, recognition_classes, args.similarity_thresh)
            base_name, suffix = os.path.splitext(image)
            result_save_path = base_name + "_recognition_result" + suffix
            print(">>>> Saving result to:", result_save_path)
            cv2.imwrite(result_save_path, dest_image)
    else:
        cosine_similarities, gathered_emb, gathered_bbs, gathered_ccs = mm.compare_images(args.images)
        print("cosine_similarities:", cosine_similarities, "\ngathered_bbs:", gathered_bbs, "\ngathered_ccs:", gathered_ccs)
