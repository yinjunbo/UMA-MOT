#!/usr/bin/python
# -*- coding:utf8 -*-
# vim: expandtab:ts=4:sw=4

import os
import run_public
import config.config as CONFIG
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data_dir = CONFIG.PATH['data_dir']
    det_dir = CONFIG.PATH['det_dir']
    trained_model = os.path.join(os.getcwd(), CONFIG.MODEL_DIR)
    date_format_localtime = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_name = CONFIG.NAME['save_name'] + date_format_localtime

    display = CONFIG.DISPLAY
    context_amount = CONFIG.PRAM['context_amount']
    occlusion_thres = CONFIG.PRAM['occlusion_thres']
    association_thres = CONFIG.PRAM['association_thres']
    iou = CONFIG.PRAM['iou']

    output_dir = os.path.join(os.getcwd(), CONFIG.PATH['output_dir'], save_name)
    os.makedirs(output_dir, exist_ok=True)

    sequences = os.listdir(data_dir)
    sequence_speed = []
    sequences = CONFIG.NAME['sequences'][CONFIG.TEST_TYPE]
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(data_dir, sequence)
        output_file = os.path.join(output_dir, "%s.txt" % sequence)
        info_filename = os.path.join(sequence_dir, "seqinfo.ini")
        if os.path.exists(info_filename):
            with open(info_filename, "r") as f:
                line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
                info_dict = dict(
                    s for s in line_splits if isinstance(s, list) and len(s) == 2)
                frame_count = int(info_dict["seqLength"])
                max_age = int(CONFIG.PRAM['life_span'] * int(info_dict["frameRate"]))   # max span-life or not
        else:
            img_list = os.listdir(os.path.join(sequence_dir, 'img1'))
            frame_count = int(max(img_list).split('.')[0])
        sequence_speed.append(run_public.run(
            sequence_dir, det_dir, trained_model, output_file,
            max_age,  context_amount, iou, occlusion_thres,
            association_thres, display))

    print("Runtime: %g ms, %g fps.\n%s" % (sum(sequence_speed)/len(sequence_speed), 1000/(sum(sequence_speed)/len(sequence_speed)),output_dir))

