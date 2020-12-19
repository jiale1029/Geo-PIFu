# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
from os.path import join

import subprocess
from urllib.request import Request, urlopen
# import urllib

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
    # with urllib.urlopen(url) as response:
        return response.read().decode().strip().split('\n')


def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = f'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    # cmd = ['aria2c', '-x8', url, '-d', out_path]
    print('Downloading', category, set_name, 'set')
    return f"{url} "
    # subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='')
    parser.add_argument('-c', '--category', default=None)
    args = parser.parse_args()

    categories = list_categories()
    if args.category is None:
        urls = []
        print('Downloading', len(categories), 'categories')
        for category in categories:
            urls.append(download(args.out_dir, category, 'train'))
            urls.append(download(args.out_dir, category, 'val'))
        urls.append(download(args.out_dir, '', 'test'))
        with open("urls.txt", "w") as f:
            for url in urls:
                f.write(url + "\n")
    else:
        if args.category == 'test':
            download(args.out_dir, '', 'test')
        elif args.category not in categories:
            print('Error:', args.category, "doesn't exist in", 'LSUN release')
        else:
            download(args.out_dir, args.category, 'train')
            download(args.out_dir, args.category, 'val')


if __name__ == '__main__':
    main()
