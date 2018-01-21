# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:09:19 2018

@author: dev1
"""

import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# SSD（差の二乗和）を計算する関数
def compute_score_map(template, target):
    th, tw = template.shape

    score_map = np.zeros(shape=(target.shape[0] - th,
                                target.shape[1] - tw))
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):
            diff = target[y:y + th, x:x + tw] - template
            score_map[y, x] = np.square(diff).sum()

    return score_map


def main():
    # 要書き換え
    target_path = r'C:\work\py\mandrill\mandrill2.png'
    template_path = r'C:\work\py\mandrill\target.png'
    template = io.imread(template_path, as_grey=True)
    target = io.imread(target_path, as_grey=True)
    th, tw = template.shape

    # スケールを変えながら差の二乗和を計算して配列score_mapsに格納
    score_maps = []
    scale_factor = 2.0 ** (-1.0 / 8.0)
    target_scaled = target + 0
    for s in range(8):
        score_maps.append(compute_score_map(template, target_scaled))
        target_scaled = transform.rescale(target_scaled, scale_factor,
                                          mode='constant')

    # 配列に格納されているnp.arrrayからSSDが最小の物を抜き出す
    # お約束の書き方として覚えた方が良いかも
    score, s, (x, y) = min([(
                np.min(score_map),
                s,
                np.unravel_index(np.argmin(score_map), score_map.shape))
                for s, score_map in enumerate(score_maps)
                ])

    # 画像に展開
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
    ax1.imshow(template, cmap=cm.Greys_r)
    ax1.set_axis_off()
    ax1.set_title('template')
    ax2.imshow(target, cmap=cm.Greys_r)
    ax2.set_title('target')
    scale = (scale_factor ** s)
    th, tw = template.shape
    rect = plt.Rectangle((y / scale, x / scale), tw/scale,
                         th/scale, edgecolor='r', facecolor='none')

    ax2.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    main()
