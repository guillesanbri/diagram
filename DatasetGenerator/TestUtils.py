import unittest
import utils
import numpy as np


class TestUtils(unittest.TestCase):
    def test_scale_image_up_1(self):
        img_mock = np.zeros((200, 150), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 400).shape, (400, 300))

    def test_scale_image_up_2(self):
        img_mock = np.zeros((10, 251), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 1024).shape, (40, 1024))

    def test_scale_image_up_3(self):
        img_mock = np.zeros((37, 37), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 100).shape, (100, 100))

    def test_scale_image_down_1(self):
        img_mock = np.zeros((1024, 1000), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 512).shape, (512, 500))

    def test_scale_image_down_2(self):
        img_mock = np.zeros((1000, 2048), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 600).shape, (292, 600))

    def test_scale_image_down_3(self):
        img_mock = np.zeros((1024, 1024), dtype=np.uint8)
        self.assertEqual(utils.scale_image(img_mock, 25).shape, (25, 25))

    def test_scale_image_shape_less_2(self):
        img_mock = np.ones(1024, dtype=np.uint8)
        self.assertRaises(ValueError, utils.scale_image, img_mock, 25)

    def test_scale_image_shape_more_2(self):
        img_mock = np.ones((1024, 1024, 3), dtype=np.uint8)
        self.assertRaises(ValueError, utils.scale_image, img_mock, 25)

    def test_scale_image_assert_binary_up(self):
        # Assert there is no value other than 0 or 255 after interpolation.
        img_mock = np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)*255
        resized = utils.scale_image(img_mock, 2048)
        resized[resized == 255] = 0
        self.assertEqual(np.sum(resized), 0)

    def test_scale_image_assert_binary_down(self):
        # Assert there is no value other than 0 or 255 after interpolation.
        img_mock = np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)*255
        resized = utils.scale_image(img_mock, 100)
        resized[resized == 255] = 0
        self.assertEqual(np.sum(resized), 0)


if __name__ == '__main__':
    unittest.main()
