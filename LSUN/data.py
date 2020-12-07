from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join
import pdb # pdb.set_trace()

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break

def export_images(db_path, out_dir, flat=False, limit=-1, imageType='jpg'):
    
    print('Exporting', db_path, 'to', out_dir)
    sceneName = db_path.split("/")[-1].replace("_lmdb","").replace("_val","").replace("_train","")

    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:

            if imageType == "webp":

                if not flat:
                    image_out_dir = join(out_dir, '/'.join(key[:6]))
                else:
                    image_out_dir = out_dir

                if not exists(image_out_dir):
                    os.makedirs(image_out_dir)
                image_out_path = join(image_out_dir, key + '.webp')
                with open(image_out_path, 'w') as fp:
                    fp.write(val)

            elif imageType == "jpg":
                
                if not flat:
                    image_out_dir = join(out_dir, sceneName)
                else:
                    image_out_dir = out_dir

                if not exists(image_out_dir):
                    os.makedirs(image_out_dir)

                img = cv2.imdecode(numpy.fromstring(val, dtype=numpy.uint8), 1)

                image_out_path = join(image_out_dir, key + '.jpg')
                cv2.imwrite(image_out_path,img)

            else:
                print("imageType: %s is not defined!" % (imageType))
                pdb.set_trace()

            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')

    # save dataset statistics
    countTxtPath = "%s/count.txt" % (out_dir)
    openMode = 'a' if os.path.exists(countTxtPath) else 'w'
    with open(countTxtPath, openMode) as fp:
        fp.write("%s %d\n" % (sceneName, count))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--flat', action='store_true',
                        help='If enabled, the images are imported into output '
                             'directory directly instead of hierarchical '
                             'directories.')
    parser.add_argument('--imageType', type=str, default='')
    parser.add_argument('--limit', type=int, default='-1')
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path
    for lmdb_path in lmdb_paths:
        if command == 'view':
            view(lmdb_path)
        elif command == 'export':
            export_images(lmdb_path, args.out_dir, args.flat, limit=args.limit, imageType=args.imageType)

if __name__ == '__main__':
    main()
