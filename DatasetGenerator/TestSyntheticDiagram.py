import unittest
from SyntheticDiagram import SyntheticDiagram


class TestSyntheticDiagram(unittest.TestCase):
    def test_overlapping_b1_over_b2(self):
        b1 = {"ulx": 0, "uly": 0, "lrx": 100, "lry": 100}
        b2 = {"ulx": 0, "uly": 110, "lrx": 100, "lry": 210}
        overlap = SyntheticDiagram.overlaps(b1, b2)
        self.assertFalse(overlap)

    def test_overlapping_b2_over_b1(self):
        b2 = {"ulx": 0, "uly": 0, "lrx": 100, "lry": 100}
        b1 = {"ulx": 0, "uly": 110, "lrx": 100, "lry": 210}
        overlap = SyntheticDiagram.overlaps(b1, b2)
        self.assertFalse(overlap)

    def test_overlapping_b1_right_b2(self):
        b1 = {"ulx": 0, "uly": 0, "lrx": 100, "lry": 100}
        b2 = {"ulx": -100, "uly": 0, "lrx": -10, "lry": 100}
        overlap = SyntheticDiagram.overlaps(b1, b2)
        self.assertFalse(overlap)

    def test_overlapping_b2_right_b1(self):
        b2 = {"ulx": 0, "uly": 0, "lrx": 100, "lry": 100}
        b1 = {"ulx": -100, "uly": 0, "lrx": -10, "lry": 100}
        overlap = SyntheticDiagram.overlaps(b1, b2)
        self.assertFalse(overlap)

    def test_overlapping_true(self):
        b1 = {"ulx": 0, "uly": 0, "lrx": 100, "lry": 100}
        b2 = {"ulx": 50, "uly": 50, "lrx": 150, "lry": 150}
        overlap = SyntheticDiagram.overlaps(b1, b2)
        self.assertTrue(overlap)


if __name__ == '__main__':
    unittest.main()
