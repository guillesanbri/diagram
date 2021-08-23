from DatasetGenerator.ElementExtractor import ElementExtractor
import unittest


# TODO: Fix tests
class TestElementExtractor(unittest.TestCase):

    def test_init_relative_path(self):
        ee = ElementExtractor("tests/m12-34w-x1x-yyy-zzz.jpeg")
        self.assertEqual(ee.sheet_id, "m12")
        self.assertEqual(ee.author_id, "34w")
        self.assertEqual(ee.device_id, "x1x")
        self.assertEqual(ee.element_id, "yyy")
        self.assertEqual(ee.tool_id, "zzz")

    def test_init_path_extension(self):
        ee = ElementExtractor("tests/moc-34w-x1x-yyy-zzz.png")
        self.assertEqual(ee.sheet_id, "moc")
        self.assertEqual(ee.author_id, "34w")
        self.assertEqual(ee.device_id, "x1x")
        self.assertEqual(ee.element_id, "yyy")
        self.assertEqual(ee.tool_id, "zzz")

    def test_init_name_format_error(self):
        self.assertRaises(ValueError, ElementExtractor, "lenna.jpeg")

    def test_init_image_error(self):
        self.assertRaises(FileNotFoundError, ElementExtractor, "sss-34w-x1x-yyy-zzz.png")


if __name__ == '__main__':
    unittest.main()
