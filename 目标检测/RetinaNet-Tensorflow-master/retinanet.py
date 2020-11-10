import tensorflow as tf
import math
from normalization import Normalization
import resnet
import densenet
import mobilenet_v2
from model import Model, Sequential


# TODO: refactor with tf.layers.Layer?

def build_backbone(backbone, activation, dropout_rate):
    assert backbone in ['resnet_50', 'densenet_121', 'densenet_169', 'mobilenet_v2']
    if backbone == 'resnet_50':
        return resnet.ResNeXt_50(activation=activation)
    elif backbone == 'densenet_121':
        return densenet.DenseNetBC_121(activation=activation, dropout_rate=dropout_rate)
    elif backbone == 'densenet_169':
        return densenet.DenseNetBC_169(activation=activation, dropout_rate=dropout_rate)
    elif backbone == 'mobilenet_v2':
        return mobilenet_v2.MobileNetV2(activation=activation, dropout_rate=dropout_rate)


class ClassificationSubnet(Model):
    def __init__(self,
                 num_anchors,
                 num_classes,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.pre_conv = Sequential([
            Sequential([
                tf.layers.Conv2D(
                    256,
                    3,
                    1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                Normalization(),
                activation,
            ]) for _ in range(4)
        ])

        pi = 0.01
        bias_prior_initializer = tf.constant_initializer(-math.log((1 - pi) / pi))

        self.out_conv = tf.layers.Conv2D(
            num_anchors * num_classes,
            3,
            1,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_prior_initializer)

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

        return input


class RegressionSubnet(Model):
    def __init__(self,
                 num_anchors,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors

        self.pre_conv = Sequential([
            Sequential([
                tf.layers.Conv2D(
                    256,
                    3,
                    1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                Normalization(),
                activation,
            ]) for _ in range(4)
        ])

        self.out_conv = tf.layers.Conv2D(
            num_anchors * 4,
            3,
            1,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, 4))

        return input


class FeaturePyramidNetwork(Model):
    class UpsampleMerge(Model):
        def __init__(self,
                     kernel_initializer,
                     kernel_regularizer,
                     name='upsample_merge'):
            super().__init__(name=name)

            self.conv_lateral = Sequential([
                tf.layers.Conv2D(
                    256,
                    1,
                    1,
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                Normalization()
            ])

            self.conv_merge = Sequential([
                tf.layers.Conv2D(
                    256,
                    3,
                    1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                Normalization()
            ])

        # TODO: refactor arguments
        # TODO: refactor upsampling to function
        def call(self, lateral, downsampled, training):
            lateral = self.conv_lateral(lateral, training)
            lateral_size = tf.shape(lateral)[1:3]
            downsampled = tf.image.resize_images(
                downsampled, lateral_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

            merged = lateral + downsampled
            merged = self.conv_merge(merged, training)

            return merged

    def __init__(self,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 name='feature_pyramid_network'):
        super().__init__(name=name)

        self.p6_from_c5 = Sequential([
            tf.layers.Conv2D(
                256,
                3,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p7_from_p6 = Sequential([
            activation,
            tf.layers.Conv2D(
                256,
                3,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p5_from_c5 = Sequential([
            tf.layers.Conv2D(
                256,
                1,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p4_from_c4p5 = FeaturePyramidNetwork.UpsampleMerge(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='upsample_merge_c4p5')
        self.p3_from_c3p4 = FeaturePyramidNetwork.UpsampleMerge(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='upsample_merge_c3p4')

    def call(self, input, training):
        P6 = self.p6_from_c5(input['C5'], training)
        P7 = self.p7_from_p6(P6, training)
        P5 = self.p5_from_c5(input['C5'], training)
        P4 = self.p4_from_c4p5(input['C4'], P5, training)
        P3 = self.p3_from_c3p4(input['C3'], P4, training)

        return {'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7}


class RetinaNetBase(Model):
    def __init__(self,
                 backbone,
                 levels,
                 num_classes,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='retinanet_base'):
        super().__init__(name=name)

        self.backbone = build_backbone(backbone, activation=activation, dropout_rate=dropout_rate)

        if backbone == 'densenet':
            # TODO: check if this is necessary
            # DenseNet has preactivation architecture,
            # so we need to apply activation before passing features to FPN
            self.postprocess_bottom_up = {
                cn: Sequential([
                    Normalization(),
                    activation
                ])
                for cn in ['C3', 'C4', 'C5']
            }
        else:
            self.postprocess_bottom_up = None

        self.fpn = FeaturePyramidNetwork(
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.classification_subnet = ClassificationSubnet(
            num_anchors=levels.num_anchors,  # TODO: level anchor boxes
            num_classes=num_classes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='classification_subnet')

        self.regression_subnet = RegressionSubnet(
            num_anchors=levels.num_anchors,  # TODO: level anchor boxes
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='regression_subnet')

    def call(self, input, training):
        bottom_up = self.backbone(input, training)

        if self.postprocess_bottom_up is not None:
            bottom_up = {
                cn: self.postprocess_bottom_up[cn](bottom_up[cn], training)
                for cn in ['C3', 'C4', 'C5']
            }

        top_down = self.fpn(bottom_up, training)

        classifications = {
            k: self.classification_subnet(top_down[k], training)
            for k in top_down
        }

        regressions = {
            k: self.regression_subnet(top_down[k], training)
            for k in top_down
        }

        return {
            'classifications': classifications,
            'regressions': regressions
        }


class RetinaNet(Model):
    def __init__(self, backbone, levels, num_classes, activation, dropout_rate, name='retinanet'):
        super().__init__(name=name)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.base = RetinaNetBase(
            backbone=backbone,
            levels=levels,
            num_classes=num_classes,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        return self.base(input, training)
