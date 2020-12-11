# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tao dataset."""

import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.video.tao import tao


class TaoTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for tao dataset."""
  DATASET_CLASS = tao.Tao
  SPLITS = {
      tfds.Split.TRAIN: 1,
      tfds.Split.VALIDATION: 1,
      tfds.Split.TEST: 1,
  }

  def _download_and_prepare_as_dataset(self, builder):
    super()._download_and_prepare_as_dataset(builder)

    if not tf.executing_eagerly():  # Only test the following in eager mode.
      return

    with self.subTest('check_annotations'):
      splits = builder.as_dataset()
      train_ex = list(splits[tfds.Split.TRAIN])[0]
      val_ex = list(splits[tfds.Split.VALIDATION])[0]
      test_ex = list(splits[tfds.Split.TEST])[0]
      # The test example has no labels.
      self.assertEmpty(test_ex['annotations']['label_frame_indices'])

      for ex in [train_ex, val_ex]:
        # There should be the same number of each of these; a number
        # per group of bboxes indicating which frame they correspond to.
        self.assertEqual(
            ex['annotations']['per_image_bboxes']['bboxes'].shape[0],
            ex['annotations']['label_frame_indices'].shape[0])
      # Check that each bbox in each image has an is_crowd and class label.
      for anno_idx, bboxes in enumerate(
          ex['annotations']['per_image_bboxes']['bboxes']):
        is_crowds = ex['annotations']['per_image_bboxes']['is_crowds'][anno_idx]
        classes = ex['annotations']['per_image_bboxes']['classes'][anno_idx]
        self.assertEqual(len(bboxes), len(classes))
        self.assertEqual(len(bboxes), len(is_crowds))

    with self.subTest('check_video'):
      splits = builder.as_dataset()
      train_ex = list(splits[tfds.Split.TRAIN])[0]
      val_ex = list(splits[tfds.Split.VALIDATION])[0]
      test_ex = list(splits[tfds.Split.TEST])[0]
      # NOTE: For real images, this will be a list of potentially a thousand or
      # more frames. For testing purposes we load a single dummy 10 X 10 image.
      self.assertEqual(train_ex['video']['frames'].shape, (1, 10, 10, 3))
      self.assertEqual(val_ex['video']['frames'].shape, (1, 10, 10, 3))
      self.assertEqual(test_ex['video']['frames'].shape, (1, 10, 10, 3))

if __name__ == '__main__':
  tfds.testing.test_main()
