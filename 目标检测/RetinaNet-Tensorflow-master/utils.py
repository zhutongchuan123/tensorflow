import termcolor
import tensorflow as tf
import cv2
from tensorflow.python.client import device_lib
import numpy as np
from collections import namedtuple
from typing import List

NMS_MAX_OUTPUT_SIZE = 1000
BoxesDecoded = namedtuple('BoxesDecoded', ['boxes', 'scores', 'class_ids'])
ClassmapDecoded = namedtuple('ClassmapDecoded', ['fg_mask'])
Detection = namedtuple('Detection', ['classification', 'regression', 'regression_postprocessed'])
Classification = namedtuple('Classification', ['unscaled', 'prob'])


def log_args(args):
    print(termcolor.colored('arguments:', 'yellow'))
    for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
        print(termcolor.colored('\t{}:'.format(key), 'yellow'), value)


def boxmap_anchor_relative_to_image_relative(regression, name='boxmap_anchor_relative_to_image_relative'):
    with tf.name_scope(name):
        grid_size = tf.shape(regression)[1:3]
        cell_size = tf.to_float(1 / grid_size)

        grid_y_pos = tf.linspace(cell_size[0] / 2, 1 - cell_size[0] / 2, grid_size[0])
        grid_x_pos = tf.linspace(cell_size[1] / 2, 1 - cell_size[1] / 2, grid_size[1])

        grid_x_pos, grid_y_pos = tf.meshgrid(grid_x_pos, grid_y_pos)
        grid_pos = tf.stack([grid_y_pos, grid_x_pos], -1)
        grid_pos = tf.expand_dims(grid_pos, -2)

        pos, size = tf.split(regression, 2, -1)

        return tf.concat([pos + grid_pos, size], -1)


def boxmap_center_relative_to_corner_relative(regression, name='boxmap_center_relative_to_corner_relative'):
    with tf.name_scope(name):
        pos = regression[..., :2]
        half_size = regression[..., 2:] / 2

        return tf.concat([pos - half_size, pos + half_size], -1)


def anchor_boxmap(grid_size, anchor_boxes, name='anchor_boxmap'):
    with tf.name_scope(name):
        num_boxes = tf.shape(anchor_boxes)[0]
        positions = tf.zeros_like(anchor_boxes)
        anchor_boxes = tf.concat([positions, anchor_boxes], -1)
        anchor_boxes = tf.reshape(anchor_boxes, (1, 1, 1, num_boxes, 4))
        anchor_boxes = tf.tile(anchor_boxes, (1, grid_size[0], grid_size[1], 1, 1))

        boxmap = boxmap_anchor_relative_to_image_relative(anchor_boxes)
        boxmap = boxmap_center_relative_to_corner_relative(boxmap)

        return boxmap


# TODO: refactor
def iou(a, b, name='iou'):
    with tf.name_scope(name):
        # TODO: should be <
        with tf.control_dependencies([
            tf.assert_less_equal(a[..., :2], a[..., 2:]),
            tf.assert_less_equal(b[..., :2], b[..., 2:])
        ]):
            # determine the coordinates of the intersection rectangle
            y_top = tf.maximum(a[..., 0], b[..., 0])
            x_left = tf.maximum(a[..., 1], b[..., 1])
            y_bottom = tf.minimum(a[..., 2], b[..., 2])
            x_right = tf.minimum(a[..., 3], b[..., 3])

        invalid_mask = tf.logical_or(y_bottom < y_top, x_right < x_left)

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (y_bottom - y_top) * (x_right - x_left)

        # compute the area of both AABBs
        box_a_area = (a[..., 2] - a[..., 0]) * (
                a[..., 3] - a[..., 1])
        box_b_area = (b[..., 2] - b[..., 0]) * (
                b[..., 3] - b[..., 1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / tf.to_float(
            box_a_area + box_b_area - intersection_area)
        iou = tf.where(invalid_mask, tf.zeros_like(iou), iou)

        with tf.control_dependencies([tf.assert_greater_equal(iou, 0.0), tf.assert_less_equal(iou, 1.0)]):
            iou = tf.identity(iou)

        return iou


def scale_regression(regression, anchor_boxes, name='scale_regression'):
    with tf.name_scope(name):
        anchor_boxes = tf.tile(anchor_boxes, (1, 2))
        anchor_boxes = tf.reshape(anchor_boxes, (1, 1, 1, anchor_boxes.shape[0], anchor_boxes.shape[1]))

        return regression * anchor_boxes


def regression_postprocess(regression, anchor_boxes, name='regression_postprocess'):
    with tf.name_scope(name):
        shifts, scales = tf.split(regression, 2, -1)
        regression = tf.concat([shifts, tf.exp(scales)], -1)

        regression = scale_regression(regression, anchor_boxes)
        regression = boxmap_anchor_relative_to_image_relative(regression)
        regression = boxmap_center_relative_to_corner_relative(regression)

        return regression


def draw_bounding_boxes(image, boxes, class_ids, class_names, font_scale=0.3):
    rng = np.random.RandomState(42)
    colors = [(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)) for _ in range(len(class_names))]

    image = np.copy(image)
    input_size = image.shape[:2]
    boxes_scale = np.array([*input_size, *input_size])  # TODO: -1 ?
    boxes = (boxes * boxes_scale).round().astype(np.int32)
    for box, class_id in zip(boxes, class_ids):
        image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), colors[class_id], 1)

        text_size, baseline = cv2.getTextSize(class_names[class_id], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        box_offset = (-text_size[1] - baseline, 0)
        text_offset = -baseline
        if box[0] + box_offset[0] < 0:
            box_offset = (0, text_size[1] + baseline)
            text_offset = text_size[1]

        image = cv2.rectangle(
            image, (box[1], box[0] + box_offset[0]), (box[1] + text_size[0], box[0] + box_offset[1]), colors[class_id],
            -1)
        text_color = (0, 0, 0) if np.mean(colors[class_id]) > 255 / 2 else (255, 255, 255)
        image = cv2.putText(
            image, class_names[class_id], (box[1], box[0] + text_offset), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color, lineType=cv2.LINE_AA)

    return image


def merge_outputs(dict, name='merge_outputs'):
    with tf.name_scope(name):
        return tf.concat(list(dict.values()), 0)


def all_same(items):
    return all(x == items[0] for x in items)


def dict_map(f, dict):
    return {k: f(dict[k]) for k in dict}


def dict_starmap(f, dicts):
    assert all_same([d.keys() for d in dicts])
    keys = dicts[0].keys()
    return {k: f(*[d[k] for d in dicts]) for k in keys}


# TODO: remove this or refactor
def classmap_decode(classmap, name='classmap_decoder'):
    with tf.name_scope(name):
        classmap_max = tf.reduce_max(classmap, -1)
        fg_mask = classmap_max > 0.5

        # scores = tf.boolean_mask(tf.reduce_max(classmap, ))
        # classmap = tf.where(fg_mask, tf.argmax(classmap, -1), tf.fill(tf.shape(fg_mask), tf.to_int64(-1)))

        return ClassmapDecoded(fg_mask=fg_mask)


# TODO: use classmap_decode
def boxes_decode(classifications, regressions, name='boxes_decode'):
    with tf.name_scope(name):
        classifications_max = tf.reduce_max(classifications, -1)
        class_ids = tf.argmax(classifications, -1)
        fg_mask = classifications_max > 0.5
        boxes = tf.boolean_mask(regressions, fg_mask)
        scores = tf.boolean_mask(classifications_max, fg_mask)
        class_ids = tf.boolean_mask(class_ids, fg_mask)

        return BoxesDecoded(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids)


def nms_classwise(decoded: BoxesDecoded, num_classes, name='nms_classwise'):
    with tf.name_scope(name):
        decoded_classwise = []
        for c in range(num_classes):
            mask = tf.equal(decoded.class_ids, c)
            suppressed = BoxesDecoded(
                boxes=tf.boolean_mask(decoded.boxes, mask),
                scores=tf.boolean_mask(decoded.scores, mask),
                class_ids=tf.boolean_mask(decoded.class_ids, mask))
            suppressed = nms(suppressed)
            decoded_classwise.append(suppressed)

        return merge_boxes_decoded(decoded_classwise)


def nms(decoded: BoxesDecoded, max_output_size=NMS_MAX_OUTPUT_SIZE, name='nms'):
    with tf.name_scope(name):
        nms_indices = tf.image.non_max_suppression(decoded.boxes, decoded.scores, max_output_size, iou_threshold=0.5)

        return BoxesDecoded(
            boxes=tf.gather(decoded.boxes, nms_indices),
            scores=tf.gather(decoded.scores, nms_indices),
            class_ids=tf.gather(decoded.class_ids, nms_indices))


def merge_boxes_decoded(decoded: List[BoxesDecoded]):
    return BoxesDecoded(
        boxes=tf.concat([d.boxes for d in decoded], 0),
        scores=tf.concat([d.scores for d in decoded], 0),
        class_ids=tf.concat([d.class_ids for d in decoded], 0))


def dict_update(dict, keys, f):
    if len(keys) == 0:
        return f(dict)

    return {
        **dict,
        keys[0]: dict_update(dict[keys[0]], keys[1:], f)
    }


def process_labels_and_logits(labels, logits, levels, name='process_labels_and_logits'):
    with tf.name_scope(name):
        labels = dict_update(
            labels,
            ['detection', 'classifications'],
            lambda c: Classification(unscaled=None, prob=c))
        logits = dict_update(
            logits,
            ['detection', 'classifications'],
            lambda c: Classification(unscaled=c, prob=dict_map(tf.nn.sigmoid, c)))

        image_size = tf.shape(labels['image'])[1:3]
        labels = postprocess_and_mask(labels, labels['trainable_masks'], image_size=image_size, levels=levels)
        logits = postprocess_and_mask(logits, labels['trainable_masks'], image_size=image_size, levels=levels)

        return labels, logits


def postprocess_and_mask(input, trainable_masks, image_size, levels, name='postprocess_and_mask'):
    with tf.name_scope(name):
        detection = Detection(
            classification=input['detection']['classifications'],
            regression=input['detection']['regressions'],
            regression_postprocessed=dict_starmap(
                lambda r, l: regression_postprocess(r, tf.to_float(l.anchor_sizes / image_size)),
                (input['detection']['regressions'], levels)))

        unscaled = detection.classification.unscaled
        prob = detection.classification.prob
        detection_trainable_classifications = Classification(
            unscaled=merge_outputs(dict_starmap(tf.boolean_mask, (unscaled, trainable_masks))) if unscaled else None,
            prob=merge_outputs(dict_starmap(tf.boolean_mask, (prob, trainable_masks))) if prob else None)

        detection_trainable = Detection(
            classification=detection_trainable_classifications,
            regression=merge_outputs(dict_starmap(
                tf.boolean_mask, (detection.regression, trainable_masks))),
            regression_postprocessed=merge_outputs(dict_starmap(
                tf.boolean_mask, (detection.regression_postprocessed, trainable_masks))))

        return {
            **input,
            'detection': detection,
            'detection_trainable': detection_trainable
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = cv2.imread('./data/tf-logo.png')
    image = cv2.resize(image, (400, 400))

    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.1, 0.6, 0.6],
        [0.25, 0.25, 0.75, 0.75],
        [0.4, 0.4, 0.9, 0.9],
    ])

    class_ids = np.array([0, 3, 6, 9])

    class_names = list('abcdefghjk')

    image = draw_bounding_boxes(image, boxes, class_ids, class_names)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def get_num_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']

    return len(gpus)
