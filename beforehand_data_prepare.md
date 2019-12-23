# ___Beforehand Data Prepare___
***

## Data source from Insightface
  - Training Data in this project is `MS1M-ArcFace` downloaded from [Insightface Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
  - Evaluating data is `LFW` `CFP-FP` `AgeDB-30` bin files included in `MS1M-ArcFace` dataset
  - Any other data is also available just in the right format
## Training dataset
  - Extract data from mxnet record format to folders like
    ```sh
    .
    ├── 0
    │   ├── 100.jpg
    │   ├── 101.jpg
    │   └── 102.jpg
    ├── 1
    │   ├── 111.jpg
    │   ├── 112.jpg
    │   └── 113.jpg
    ├── 10
    │   ├── 707.jpg
    │   ├── 708.jpg
    │   └── 709.jpg
    ```
  - **Python code** source folder is `/datasets/faces_emore`, extract to dist folder `/datasets/faces_emore_112x112_folders`, this may take hours.
    ```py
    import os
    import numpy as np
    import mxnet as mx
    from tqdm import tqdm

    # read_dir = '/datasets/faces_glint/'
    # save_dir = '/datasets/faces_glint_112x112_folders'
    read_dir = '/datasets/faces_emore'
    save_dir = '/datasets/faces_emore_112x112_folders'
    idx_path = os.path.join(read_dir, 'train.idx')
    bin_path = os.path.join(read_dir, 'train.rec')

    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
    rec_header, _ = mx.recordio.unpack(imgrec.read_idx(0))

    # for ii in tqdm(range(1, 10)):
    for ii in tqdm(range(1, int(rec_header.label[0]))):
        img_info = imgrec.read_idx(ii)
        header, img = mx.recordio.unpack(img_info)
        img_idx = str(int(np.sum(header.label)))
        img_save_dir = os.path.join(save_dir, img_idx)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        # print(os.path.join(img_save_dir, str(ii) + '.jpg'))
        with open(os.path.join(img_save_dir, str(ii) + '.jpg'), 'wb') as ff:
            ff.write(img)
    ```
## Evaluting bin files
  - Bin files include jpeg image data pairs, and a label indicating if it's a same person, so there are double images than labels
    ```sh
    # bins | issame_list
    img_1 img_2 | 1
    img_3 img_4 | 1
    img_5 img_6 | 0
    img_7 img_8 | 0
    ```
  - Image data in bin files like `CFP-FP` `AgeDB-30` is not compatible with `tf.image.decode_jpeg`, we need to reformat it.
  - **Python code**
    ```py
    import io
    import pickle
    import tensorflow as tf
    from skimage.io import imread

    test_bin_files = ['/datasets/faces_emore/agedb_30.bin', '/datasets/faces_emore/cfp_fp.bin']
    for test_bin_file in test_bin_files:
        with open(test_bin_file, 'rb') as ff:
            bins, issame_list = pickle.load(ff, encoding='bytes')

        bb = [tf.image.encode_jpeg(imread(io.BytesIO(ii))) for ii in bins]
        with open(test_bin_file, 'wb') as ff:
            pickle.dump([bb, issame_list], ff)
    ```
## Conclusion
  - After these steps, we made our training dataset folder and evaluating dataset bin files
***
