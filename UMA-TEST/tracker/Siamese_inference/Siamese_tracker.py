import logging
import numpy as np
from tracker.Siamese_utils.infer_utils import convert_bbox_format, Rectangle
from tracker.Siamese_utils.misc_utils import get_center


class TargetState(object):
  """Represent the target state."""

  def __init__(self, bbox, search_pos, scale_idx, his_feature, original_search_center, original_target_wh, init_templates, reid_templates, similarity, bbox_in):
    self.bbox = bbox  # (cx, cy, w, h) in the original image
    self.search_pos = search_pos  # target center position in the search image
    self.scale_idx = scale_idx  # scale index in the searched scales
    self.original_search_center = original_search_center
    self.original_target_wh = original_target_wh
    self.init_templates = init_templates
    self.reid_templates = reid_templates
    self.similarity = similarity
    self.bbox_in = bbox_in
    self.his_feature = his_feature
    self.old_bbox = bbox
    self.old_scale_idx = scale_idx
    self.old_search_pos = search_pos


class Siamese_Tracker(object):
  """Tracker based on the siamese model."""

  def __init__(self, siamese_model, model_config, track_config):
    self.siamese_model = siamese_model
    self.model_config = model_config
    self.track_config = track_config

    self.num_scales = track_config['num_scales']
    logging.info('track num scales -- {}'.format(self.num_scales))
    scales = np.arange(self.num_scales) - get_center(self.num_scales)
    self.search_factors = [self.track_config['scale_step'] ** x for x in scales]   #0.963, 1, 1.0375
    self.x_image_size = track_config['x_image_size']  # Search image size 255
    self.window = None  # Cosine window
    self.log_level = track_config['log_level']

  def init_tracks(self, sess, det, filename):

    # Get initial target bounding box and convert to center based
    init_bb = Rectangle(int(det[0]) - 1, int(det[1]) - 1, int(det[2]), int(det[3]))
    bbox = convert_bbox_format(init_bb, 'center-based')

    # Feed in the first frame image to set initial state.
    bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
    bbox_in = [init_bb.x, init_bb.y, init_bb.width, init_bb.height]
    input_feed = [filename, bbox_feed]
    templates, reid_templates = self.siamese_model.initialize(sess, input_feed)

    his_feature=[]
    his_feature.append(reid_templates)

    search_center = np.array([get_center(self.x_image_size),
                              get_center(self.x_image_size)])
    current_target_state = TargetState(bbox=bbox,
                                       search_pos=search_center,
                                       original_search_center=search_center,
                                       scale_idx=int(get_center(self.num_scales)),
                                       original_target_wh=[bbox.width, bbox.height],
                                       init_templates=templates, # used for SOT
                                       his_feature=his_feature,  # used for re-id
                                       reid_templates=reid_templates,
                                       similarity=1.0,
                                       bbox_in=bbox_in,
                                       )   # bbox_in  [x,y,w,h]
    return current_target_state

  def track(self, sess, current_target_state, filename):
    """Runs tracking on a single image sequence."""

    def roi_align(image, disp_instance_feat, height, width):
      """
      `image` is a 3-D array, representing the input feature map
      `disp_instance_feat` box center
      `height` and `width` are the desired spatial size of output feature map
      """
      crop_center = disp_instance_feat + get_center(image.shape[0])
      crop_box = [np.maximum(crop_center[0]-3, 0), np.maximum(crop_center[1]-3, 0),
                  np.minimum(crop_center[0] + 3, image.shape[0]),
                  np.minimum(crop_center[1] + 3, image.shape[0])]
      if (int(crop_box[2]-crop_box[0]) != 6) or (int(crop_box[3]-crop_box[1]) != 6):  # pad if reach boundary
        image = np.pad(image, ((6, 6), (6, 6), (0, 0)), 'constant', constant_values=np.mean(image))
        crop_center = crop_center + 6
        crop_box = [crop_center[0] - 3, crop_center[1] - 3, crop_center[0] + 3, crop_center[1] + 3]

      crop_box = [ele/image.shape[0] for ele in crop_box]

      y_min, x_min, y_max, x_max = crop_box

      img_height, img_width, channel_num = image.shape

      feature_map = []

      for y in np.linspace(y_min, y_max, height) * (img_height - 1):
        for x in np.linspace(x_min, x_max, width) * (img_height - 1):
          y_l, y_h = np.floor(y).astype('int32'), np.ceil(y).astype('int32')
          x_l, x_h = np.floor(x).astype('int32'), np.ceil(x).astype('int32')

          a = image[y_l, x_l]
          b = image[y_l, x_h]
          c = image[y_h, x_l]
          d = image[y_h, x_h]

          y_weight = y - y_l
          x_weight = x - x_l

          val = a * (1 - x_weight) * (1 - y_weight) + \
                b * x_weight * (1 - y_weight) + \
                c * y_weight * (1 - x_weight) + \
                d * x_weight * y_weight

          feature_map.append(val)

      return np.array(feature_map).reshape(height, width, channel_num)

    def roi_crop(disp_instance_feat, instance):
      instance_pad = instance.copy()
      crop_center = np.round(disp_instance_feat + get_center(instance_size)).astype(int)
      crop_box = [np.maximum(crop_center[0]-3, 0), np.maximum(crop_center[1]-3, 0),
                  np.minimum(crop_center[0] + 3, instance_size),
                  np.minimum(crop_center[1] + 3, instance_size)]
      if (int(crop_box[2]-crop_box[0]) != 6) or (int(crop_box[3]-crop_box[1]) != 6):  # padding if reach border
        instance_pad = np.pad(instance_pad, ((6, 6), (6, 6), (0, 0)), 'constant', constant_values=np.mean(instance_pad))
        crop_center = crop_center + 6
        crop_box = [crop_center[0] - 3, crop_center[1] - 3, crop_center[0] + 3, crop_center[1] + 3]
        # print(crop_box)
      instance_crop = instance_pad[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3], :]
      return instance_crop

    def npair_distance(a, b, data_is_normalized=False):
        b = np.vstack(b)
        a = np.reshape(a, (1, -1))
        b = np.reshape(b, (b.shape[0], -1))
        if not data_is_normalized:
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        return np.mean(np.dot(a, b.T))

    current_target_state.old_bbox = current_target_state.bbox   # [x_c,y_c,w,h]
    current_target_state.old_scale_idx = current_target_state.scale_idx
    current_target_state.old_search_pos = current_target_state.search_pos

    bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                 current_target_state.bbox.height, current_target_state.bbox.width]  # center x y
    bbox_feed_ltwh = [current_target_state.bbox.x - current_target_state.bbox.width/2,
                      current_target_state.bbox.y - current_target_state.bbox.height/2,
                 current_target_state.bbox.width, current_target_state.bbox.height]

    templates = current_target_state.init_templates
    input_feed = [filename, bbox_feed, templates]
    outputs = self.siamese_model.inference_step(sess, input_feed)
    search_scale_list = outputs['scale_xs']
    response = outputs['response_up']  # [3,272,272]
    instance = outputs['instance']  # [3,22,22,256]
    reid_instance = outputs['instance_reid']  # [3,22,22,256]
    response_size = response.shape[1]
    instance_size = instance.shape[1]

    # Choose the scale whole response map has the highest peak
    if self.num_scales > 1:
      response_max = np.max(response, axis=(1, 2))
      penalties = self.track_config['scale_penalty'] * np.ones(self.num_scales)
      current_scale_idx = int(get_center(self.num_scales))
      penalties[current_scale_idx] = 1.0
      response_penalized = response_max * penalties
      best_scale = np.argmax(response_penalized)
    else:
      best_scale = 0

    response = response[best_scale]

    with np.errstate(all='raise'):  # Raise error if something goes wrong
      response = response - np.min(response)
      response = response / np.sum(response)

    if self.window is None:   # suppress the border
      window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                      np.expand_dims(np.hanning(response_size), 0))
      self.window = window / np.sum(window)  # normalize window
    window_influence = self.track_config['window_influence']  # 0.3
    response = (1 - window_influence) * response + window_influence * self.window
    # Find maximum response
    r_max, c_max = np.unravel_index(response.argmax(), response.shape)

    # Convert from crop-relative coordinates to frame coordinates
    p_coor = np.array([r_max, c_max])

    # displacement from the center in instance final representation (response comes from instance)
    disp_instance_final = p_coor - get_center(response_size)

    # ... in instance feature space ...
    upsample_factor = self.track_config['upsample_factor']
    disp_instance_feat = disp_instance_final / upsample_factor
    # ... Avoid empty position ...
    r_radius = int(response_size / upsample_factor / 2)
    disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)

    # ... in instance input ...
    disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
    # ... in instance original crop (in frame coordinates)
    disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
    # Position within frame in frame coordinates
    y = current_target_state.bbox.y
    x = current_target_state.bbox.x
    y += disp_instance_frame[0]
    x += disp_instance_frame[1]

    # compute the similarity
    instance_reid_crop1 = np.mean(roi_crop(disp_instance_feat, reid_instance[best_scale]), axis=(0, 1))
    similarity1 = npair_distance(instance_reid_crop1, current_target_state.his_feature)

    # instance_reid_crop2 = np.mean(roi_align(reid_instance[best_scale], disp_instance_feat, 6, 6), axis=(0, 1))
    # similarity2 = npair_distance(instance_reid_crop2, current_target_state.his_feature)

    current_target_state.similarity = similarity1

    # Target scale damping and saturation
    original_target_width = current_target_state.original_target_wh[0]
    original_target_height = current_target_state.original_target_wh[1]

    target_scale = current_target_state.bbox.height / original_target_height
    search_factor = self.search_factors[best_scale]
    scale_damp = self.track_config['scale_damp']  # damping factor for scale update
    target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
    target_scale = np.maximum(0.5, np.minimum(1.5, target_scale))

    # Some book keeping
    height = original_target_height * target_scale
    width = original_target_width * target_scale
    current_target_state.bbox = Rectangle(x, y, width, height)
    current_target_state.scale_idx = best_scale
    current_target_state.search_pos = current_target_state.original_search_center + disp_instance_input
    current_target_state.bbox_in = bbox_feed_ltwh

    assert 0 <= current_target_state.search_pos[0] < self.x_image_size, \
      'target position in feature space should be no larger than input image size'
    assert 0 <= current_target_state.search_pos[1] < self.x_image_size, \
      'target position in feature space should be no larger than input image size'

    track_bbox = convert_bbox_format(current_target_state.bbox, 'top-left-based')   #  center -> top left
    track_bbox = np.array([track_bbox.x, track_bbox.y, track_bbox.width, track_bbox.height])

    return current_target_state, track_bbox
