# UMA-MOT

## Introduction

This repository provides an implementation of *(CVPR20) A Unified Object Motion and Affinity Model for Online Multi-Object Tracking* (UMA-MOT).
The work integrates single object tracking and metric learning into a unified triplet network by means of multi-task learning.
Please refer the [paper](https://arxiv.org/pdf/2003.11291.pdf) for the full details.

## Requirement

* python3
* tensorflow-gpu==1.15.0

## Testing

1. Clone this repo and install dependencies
```
pip3 install -r requirements.txt
```
2. Modify `config/config.py` to add the data path. 

3. Run the inference code on MOT16 or MOT17 benchmarks. 
```
cd UMA-MOT/UMA-TEST
python3 test.py
```
4. Refer [py-motmetrics](https://github.com/cheind/py-motmetrics.git) for evaluating the tracking results in `UMA-TEST/outputs`.
```
cd UMA-MOT/motmetrics
python3 -m motmetrics.apps.eval_motchallenge DataPath/MOT-Challenge/MOT16/train ~/UMA-MOT/UMA-TEST/outputs/MOT16/MOT16_train-occ_0.8-ass_0.7-npair0.1-id0.1-se_block2-20200729_220301
```
5. Visualization.
```
cd UMA-MOT/application_util
python3 show_results.py \
       --sequence_dir=/home/junbo/datasets/MOT-Challenge/MOT16/train/MOT16-09 \
       --result_file=output_path/MOT16-09.txt \
       --detection_file=UMA-MOT/UMA-TEST/filtered_detections/MOT16-train
```

## Training 

Will be releasing.

## Results on the MOT16 train set.

```
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-04 63.8% 84.4% 51.3% 55.2% 90.8%  83  18  39  26 2653 21315  43   330 49.5% 0.214  20  14   4
MOT16-11 66.2% 83.3% 55.0% 61.8% 93.6%  69  17  26  26  385  3504  35    71 57.2% 0.226  16  18   8
MOT16-05 57.0% 81.0% 43.9% 48.2% 89.0% 125  21  63  41  407  3529  37   135 41.7% 0.257  30  19  20
MOT16-13 48.0% 83.1% 33.7% 36.8% 90.7% 107  19  38  50  431  7235  27   176 32.8% 0.283  27  17  20
MOT16-10 65.6% 88.0% 52.2% 53.9% 90.8%  54  12  28  14  673  5682  35   227 48.1% 0.262  17  13   7
MOT16-09 70.5% 84.5% 60.6% 67.5% 94.2%  25  11  13   1  219  1706  27    56 62.9% 0.265  15  10   4
MOT16-02 44.3% 81.3% 30.4% 33.4% 89.0%  54   7  22  25  734 11884  24   169 29.1% 0.255  18   9   7
OVERALL  59.9% 84.1% 46.5% 50.3% 91.0% 517 105 229 183 5502 54855 228  1164 45.1% 0.236 143 100  70
```




## Citation

If you find this project helpful in your research, please consider citing the following paper:

    @inproceedings{yin2020unified,
      title={A Unified Object Motion and Affinity Model for Online Multi-Object Tracking},
      author={Yin, Junbo and Wang, Wenguan and Meng, Qinghao and Yang, Ruigang and Shen, Jianbing},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2020},
    }




## Acknowledgement
* [SiamFC](https://github.com/bilylee/SiamFC-TensorFlow) 
* [Deep-SORT](https://github.com/nwojke/deep_sort)
