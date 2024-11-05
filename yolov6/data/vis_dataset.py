# coding=utf-8
# Description:  visualize yolo label image.

import argparse
import os
import cv2
import numpy as np

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])


def main(args):
    img_dir, label_dir, class_names = args.img_dir, args.label_dir, args.class_names

    label_map = dict()
    for class_id, classname in enumerate(class_names):
        label_map[class_id] = classname

    for file in os.listdir(img_dir):
        if file.split('.')[-1] not in IMG_FORMATS:
            print(f'[Warning]: Non-image file {file}')
            continue
        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file[: file.rindex('.')] + '.txt')

        try:
            img_data = cv2.imread(img_path)
            height, width, _ = img_data.shape
            color = [tuple(np.random.choice(range(256), size=3)) for i in class_names]
            color_poly = [tuple(np.random.choice(range(256), size=3)) for i in class_names]
            thickness = 2
            thickness_poly = 5

            with open(label_path, 'r') as f:
                for bbox in f:                        
                    cls, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_tl, y_tl, x_br, y_br = [float(v) if i > 0 else int(v) for i, v in enumerate(bbox.split('\n')[0].split(' '))]
                    # scale by image size
                    x_tl = int(x_tl * width)
                    y_tl = int(y_tl * height)
                    x_br = int(x_br * width)
                    y_br = int(y_br * height)
                    x_1 = int(x_1 * width)
                    y_1 = int(y_1 * height)
                    x_2 = int(x_2 * width)
                    y_2 = int(y_2 * height)
                    x_3 = int(x_3 * width)
                    y_3 = int(y_3 * height)
                    x_4 = int(x_4 * width)
                    y_4 = int(y_4 * height)

                    # draw rectangle based on 2 points
                    cv2.rectangle(img_data, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in color[cls]]), thickness)
                    # draw polygon based on 4 points
                    cv2.line(img_data, (x_1, y_1), (x_2, y_2), tuple([int(x) for x in color_poly[cls]]), thickness_poly)
                    cv2.line(img_data, (x_2, y_2), (x_3, y_3), tuple([int(x) for x in color_poly[cls]]), thickness_poly)
                    cv2.line(img_data, (x_3, y_3), (x_4, y_4), tuple([int(x) for x in color_poly[cls]]), thickness_poly)
                    cv2.line(img_data, (x_4, y_4), (x_1, y_1), tuple([int(x) for x in color_poly[cls]]), thickness_poly)
                    cv2.putText(img_data, label_map[cls], (x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 1, tuple([int(x) for x in color[cls]]), thickness)

            cv2.imshow('image', img_data)
            cv2.waitKey(0)
        except Exception as e:
            print(f'[Error]: {e} {img_path}')
    print('======All Done!======')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='VOCdevkit/voc_07_12/images')
    parser.add_argument('--label_dir', default='VOCdevkit/voc_07_12/labels')
    parser.add_argument('--class_names', default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

    args = parser.parse_args()
    print(args)

    main(args)
