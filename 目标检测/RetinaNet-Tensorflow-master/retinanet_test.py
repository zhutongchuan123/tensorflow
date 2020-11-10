import tensorflow as tf
import retinanet
from levels import build_levels


class FeaturePyramidNetworkTest(tf.test.TestCase):
    def test_output_shapes(self):
        image_size = 256
        input = {}
        for i in range(1, 5 + 1):
            size = image_size // 2**i
            input['C{}'.format(i)] = tf.zeros((1, size, size, 512))

        net = retinanet.FeaturePyramidNetwork(
            kernel_initializer=None, kernel_regularizer=None)
        output = net(input, False)

        for i in range(3, 7 + 1):
            size = image_size // 2**i
            assert output['P{}'.format(i)].shape == (1, size, size, 256)


class ClassificationSubnetTest(tf.test.TestCase):
    def test_output_shapes(self):
        input = tf.zeros((1, 32, 32, 256))

        net = retinanet.ClassificationSubnet(
            num_anchors=9,
            num_classes=10,
            kernel_initializer=None,
            kernel_regularizer=None)
        output = net(input, False)

        self.evaluate(tf.global_variables_initializer())
        assert self.evaluate(output).shape == (1, 32, 32, 9, 10)


class RegressionSubnet(tf.test.TestCase):
    def test_output_shapes(self):
        input = tf.zeros((1, 32, 32, 256))

        net = retinanet.RegressionSubnet(
            num_anchors=9, kernel_initializer=None, kernel_regularizer=None)
        output = net(input, False)

        self.evaluate(tf.global_variables_initializer())
        assert self.evaluate(output).shape == (1, 32, 32, 9, 4)


class RetinaNetTest(tf.test.TestCase):
    def test_output_shapes(self):
        image_size = 256
        input = tf.zeros((1, image_size, image_size, 3))
        net = retinanet.RetinaNet(
            backbone='densenet',
            levels=build_levels(),
            num_classes=10,
            dropout_rate=0.2)
        classifications, regressions = net(input, False)

        self.evaluate(tf.global_variables_initializer())
        classifications, regressions = self.evaluate(
            [classifications, regressions])

        for i in range(3, 7 + 1):
            size = image_size // 2**i
            assert classifications['P{}'.format(i)].shape == (1, size, size, 9,
                                                              10)
            assert regressions['P{}'.format(i)].shape == (1, size, size, 9, 4)
