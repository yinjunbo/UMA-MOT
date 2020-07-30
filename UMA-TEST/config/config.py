DATA_ROOT = '/home/junbo/datasets/MOT-Challenge/'
# trained model
MODEL_NAME = 'npair0.1-id0.1-se_block2'
MODEL_DIR = 'models/' + MODEL_NAME
# inference mode
TEST_DATASET = ['MOT16', 'MOT17'][0]
TEST_TYPE = ['train', 'test'][0]
DETECTOR = ['DPM', 'FRCNN', 'SDP'][0]
ATTENTION = [None, 'se_block'][1]
DISPLAY=False
# hyper-parameterc
PRAM = {
    'occlusion_thres': 0.8,   # occlusion detection
    'association_thres': 0.7,  # associate with detection
    'iou': 0.25,
    'context_amount': 0.3,
    'life_span': 10,
}
# save name
NAME = {
    'model_name': MODEL_NAME,
    'save_name': '{0}_{1}-occ_{2}-ass_{3}-{4}-'.format(TEST_DATASET, TEST_TYPE, PRAM['occlusion_thres'], PRAM['association_thres'], MODEL_NAME),
    'backbone': 'alexnet',
    'sequences': {
        'train': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'] if TEST_DATASET == 'MOT16' else
    [ele+'-{}'.format(DETECTOR) for ele in ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']],
        'test': ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'] if TEST_DATASET == 'MOT16' else
    [ele+'-{}'.format(DETECTOR) for ele in ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']]
    }
}

PATH = {
    'data_dir': DATA_ROOT + ('{}/{}'.format(TEST_DATASET, TEST_TYPE) if TEST_DATASET == 'MOT16' else '{}/{}/{}'.format(TEST_DATASET, TEST_TYPE, DETECTOR)),
    'det_dir': 'filtered_detections/' + ('{}-{}/'.format(TEST_DATASET, TEST_TYPE) if TEST_DATASET == 'MOT16' else '{}-{}-{}'.format(TEST_DATASET, TEST_TYPE, DETECTOR)),
    'output_dir':  'outputs/{}'.format(TEST_DATASET)
}



