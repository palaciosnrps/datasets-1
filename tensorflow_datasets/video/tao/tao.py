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

"""TAO dataset."""

import collections
import json
import shutil

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_VIDEO_URL = 'https://motchallenge.net/data/'
_ANNOTATIONS_URL = 'https://github.com/TAO-Dataset/annotations/archive/v1.1.tar.gz'

_DESCRIPTION = """
The TAO dataset is a large video object detection dataset consisting of
2,907 high resolution videos and 833 object categories. Note that this dataset
requires at least 300 GB of free space to store.
"""

_CITATION = """
@article{Dave_2020,
   title={TAO: A Large-Scale Benchmark for Tracking Any Object},
   ISBN={9783030585587},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-030-58558-7_26},
   DOI={10.1007/978-3-030-58558-7_26},
   journal={Lecture Notes in Computer Science},
   publisher={Springer International Publishing},
   author={Dave, Achal and Khurana, Tarasha and Tokmakov, Pavel and Schmid, Cordelia and Ramanan, Deva},
   year={2020},
   pages={436–454}
}
"""


def _merge_categories_map(annotations_dict):
  """Some categories should be renamed into others.

  This code segment is based on their provided preprocessing API.

  Args:
    annotations_dict: a dictionary containing all the annotations
  Returns:
    merge_map: dictionary mapping from category id to merged id
  """
  merge_map = {}
  for category in annotations_dict['categories']:
    if 'merged' in category:
      for to_merge in category['merged']:
        merge_map[to_merge['id']] = category['id']
  return merge_map


class Tao(tfds.core.BeamBasedBuilder):
  """DatasetBuilder for tao dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Some TAO files (HVACS and AVA videos) must be manually downloaded because
  a login to MOT is required. Please download and extract those data following
  the instructions at https://motchallenge.net/tao_download.php

  Download this data and move the resulting .zip files to
  ~/tensorflow_datasets/downloads/manual/

  If the data requiring manual download is not present, it will be skipped over
  and only the data not requiring manual download will be used.
  """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    all_features = {
        'video':
            tfds.features.Sequence({
                # NOTE: Different videos have different resolutions.
                'frames': tfds.features.Image(shape=(None, None, 3)),
            }),
        'metadata': {
            'height':
                tf.int32,
            'width':
                tf.int32,
            'num_frames':
                tf.int32,
            'video_name':
                tf.string,
            'neg_category_ids':
                tfds.features.Tensor(shape=(None,), dtype=tf.int32),
            'not_exhaustive_category_ids':
                tfds.features.Tensor(shape=(None,), dtype=tf.int32),
            'dataset':
                tf.string,
        },
        'annotations':
            tfds.features.Sequence({
                'per_image_bboxes':
                    tfds.features.Sequence({
                        'bboxes':
                            tfds.features.BBoxFeature(),
                        'classes': tfds.features.ClassLabel(num_classes=833),
                        'is_crowds': tf.bool,
                        'track_ids': tf.int32,
                        'scale_category': tf.string,
                    }),
                # Labels do not occur for all frames. This indicates the
                # indices of the frames that have labels.
                'label_frame_indices': tf.int32,
            }),
    }

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(all_features),
        supervised_keys=None,
        homepage='https://taodataset.org/',
        citation=_CITATION,
    )

  def _maybe_prepare_manual_data(self,
                                 dl_manager: tfds.download.DownloadManager):
    # If the manual download data is present, copy it over to be with the
    # rest of the automatically downloaded data and extract it.
    has_manual_data = True
    manually_downloaded_files = [
        '1_AVA_HACS_TRAIN_fc668fecdd56042fc052071b7abb7a0b.zip',
        '2_AVA_HACS_VAL_fc668fecdd56042fc052071b7abb7a0b.zip',
        '3_AVA_HACS_TEST_fc668fecdd56042fc052071b7abb7a0b.zip',
    ]
    for file in manually_downloaded_files:
      path = dl_manager.manual_dir / file
      if not path.exists():
        has_manual_data = False
        break
      shutil.copy(str(path), str(dl_manager.download_dir))
      dl_manager.extract(dl_manager.download_dir / file)
    return has_manual_data

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    train_data = dl_manager.download_and_extract(
        _VIDEO_URL + '1-TAO_TRAIN.zip')
    val_data = dl_manager.download_and_extract(
        _VIDEO_URL + '2-TAO_VAL.zip')
    test_data = dl_manager.download_and_extract(
        _VIDEO_URL + '3-TAO_TEST.zip')
    # TODO: Currently extracting annotations is broken due to b/175417709
    annotations = dl_manager.download_and_extract(
        _ANNOTATIONS_URL
    )

    has_manual_data = self._maybe_prepare_manual_data(dl_manager)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'data_path': train_data,
                'annotations_path': annotations / 'train.json',
                'has_manual_data': has_manual_data,
                'is_test_split': False,
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'data_path': val_data,
                'annotations_path': annotations / 'validation.json',
                'has_manual_data': has_manual_data,
                'is_test_split': False,
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'data_path':
                    test_data,
                'annotations_path':
                    annotations / 'test_without_annotations.json',
                'has_manual_data':
                    has_manual_data,
                'is_test_split':
                    True,
            }),
    ]

  def _preprocess_annotations(self, annotations_path):
    # NOTE: The TAO example code preprocesses the data to group some category
    # labels together in the same way done here.
    with tf.io.gfile.GFile(annotations_path, 'r') as f:
      annotations = json.load(f)
    merge_map = _merge_categories_map(annotations)
    for ann in annotations['annotations'] + annotations['tracks']:
      ann['category_id'] = merge_map.get(ann['category_id'], ann['category_id'])
    return annotations

  def _build_annotations_index(self, annotations):
    vids = {x['id']: x for x in annotations['videos']}
    vid_to_images = collections.defaultdict(list)
    image_to_anns = collections.defaultdict(list)
    for image in annotations['images']:
      vid_to_images[image['video_id']].append(image)
    for ann in annotations['annotations']:
      image_to_anns[ann['image_id']].append(ann)
    return vids, vid_to_images, image_to_anns

  def _build_pcollection(self, pipeline, data_path, annotations_path,
                         is_test_split, has_manual_data=False):
    """Yields examples."""
    beam = tfds.core.lazy_imports.apache_beam

    annotations = self._preprocess_annotations(annotations_path)
    vids, vid_to_images, image_to_anns = self._build_annotations_index(
        annotations)

    def _process_example(video_id):
      """Generate a data example for a single video."""
      video_ann = vids[video_id]
      annotation = {}
      metadata = {}
      annotation['metadata'] = metadata
      metadata['height'] = video_ann['height']
      metadata['width'] = video_ann['width']
      metadata['neg_category_ids'] = video_ann['neg_category_ids']
      metadata['not_exhaustive_category_ids'] = video_ann[
          'not_exhaustive_category_ids']
      metadata['dataset'] = video_ann['metadata']['dataset']
      metadata['video_name'] = video_ann['name']
      frames = list((data_path / 'frames' / metadata['video_name']).iterdir())
      annotation['video'] = {'frames': frames}
      metadata['num_frames'] = len(frames)
      annotation['annotations'] = []
      if is_test_split:  # There are no labels for this video.
        return metadata['video_name'], annotation
      frame_indices = [ann['frame_index'] for ann in vid_to_images[video_id]]
      # Frame indices should be sorted.
      assert all(frame_indices[i] <= frame_indices[i + 1]
                 for i in range(len(frame_indices) - 1))
      for image in vid_to_images[video_id]:
        per_image_anno = {}
        per_image_anno['per_image_bboxes'] = []
        per_image_anno['label_frame_indices'] = image['frame_index']
        annotation['annotations'].append(per_image_anno)
        for ann in image_to_anns[image['id']]:
          per_image_anno['per_image_bboxes'].append({
              'bboxes':
                  tfds.features.BBox(
                      ymin=ann['bbox'][1] / video_ann['height'],
                      ymax=(ann['bbox'][1] + ann['bbox'][3]) /
                      video_ann['height'],
                      xmin=ann['bbox'][0] / video_ann['width'],
                      xmax=(ann['bbox'][0] + ann['bbox'][2]) /
                      video_ann['width']),
              'classes':
                  ann['category_id'],
              'is_crowds':
                  ann['iscrowd'],
              'scale_category':
                  ann['scale_category'],
              'track_ids':
                  ann['track_id'],
          })
      return metadata['video_name'], annotation

    video_ids = vids.keys()
    filtered_ids = []
    for video_id in video_ids:
      video_ann = vids[video_id]
      # These must be manually downloaded.
      if ('HACS' in video_ann['metadata']['dataset'] or 'AVA' in video_ann[
          'metadata']['dataset']) and not has_manual_data:
        continue
      filtered_ids.append(video_id)

    return (
        pipeline
        | beam.Create(filtered_ids)
        | beam.Map(_process_example)
    )

