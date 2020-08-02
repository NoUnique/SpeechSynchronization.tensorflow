# -*- coding: utf-8 -*-
"""
    LRS3 dataset from http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html

    Author : NoUnique (kofmap@gmail.com)
    Copyright 2020 NoUnique. All Rights Reserved
"""

import os
import zipfile

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@InProceedings{Afouras18d,
  author       = "Afouras, T. and Chung, J.~S. and Zisserman, A.",
  title        = "LRS3-TED: a large-scale dataset for visual speech recognition",
  booktitle    = "arXiv preprint arXiv:1809.00496",
  year         = "2018",
}
"""

_DESCRIPTION = """\
LRS3 is an audio-visual speech recognition dataset collected from in the wild videos. \
The dataset consists of thousands of spoken sentences from TED and TEDx videos. \
There is no overlap between the videos used to create the test set and \
the ones used for the pre-train and trainval sets. \
The dataset statistics are given in the table below.
"""


class Lrs3Config(tfds.core.BuilderConfig):
    def __init__(self, *, width=None, height=None, sample_rate=16000, **kwargs):
        """The parameters specifying how the dataset will be processed.
        If `width` and `height` are set, the videos
        will be rescaled to have those heights and widths (using ffmpeg).
        Args:
          width: An integer with the width or None.
          height: An integer with the height or None.
          **kwargs: Passed on to the constructor of `BuilderConfig`.
        """
        super(Lrs3Config, self).__init__(**kwargs)
        if (width is None) ^ (height is None):
            raise ValueError('Either both dimensions should be set, or none of them')
        self.width = width
        self.height = height
        self.sample_rate = sample_rate


class Lrs3(tfds.core.GeneratorBasedBuilder):
    """Lip Reading Sentences 3 (LRS3) Dataset

    A audio-visual speech recognition dataset
    collected from in the wild videos.
    """

    VERSION = tfds.core.Version('1.0.0')

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain two files: vggface2_test.tar.gz and
    vggface2_train.tar.gz.
    You need to fill a form(https://goo.gl/forms/vGZmhJaZ9LAklozz2)
    in order to get the ID and password to download the dataset
    from http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html.
    """

    BUILDER_CONFIGS = [
        Lrs3Config(
            name='Lrs3_256_16000',  # LRS3 dataset originally has 256x256 size.
            sample_rate=16000,
            description='256x256 LRS3 dataset with audio data.',
            version=VERSION,
        ),
        Lrs3Config(
            name='Lrs3_128_16000',
            description='128x128 LRS3 dataset with audio data.',
            width=128,
            height=128,
            sample_rate=16000,
            version=VERSION,
        ),
    ]

    def _info(self):
        if self.builder_config.width is not None:
            if self.builder_config.height is None:
                raise ValueError('Provide either both height and width or none.')
            ffmpeg_extra_args = (
                '-vf', 'scale={}x{}'.format(self.builder_config.height,
                                            self.builder_config.width))
        else:
            ffmpeg_extra_args = []

        video_shape = (
            None, self.builder_config.height, self.builder_config.width, 3)
        features = tfds.features.FeaturesDict({
            'video': tfds.features.Video(video_shape,
                                         ffmpeg_extra_args=ffmpeg_extra_args,
                                         encoding_format='jpeg'),
            'audio': tfds.features.Audio(sample_rate=self.builder_config.sample_rate,
                                         file_format='mp4',
                                         dtype=tf.int16),
            'sentence': tfds.features.Text(),
            'video_id': tfds.features.Text(),  # YouTube video ID consists of 11 characters
            'file_name': tfds.features.Text(),
        })
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            homepage='http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html',
            supervised_keys=('video', 'sentence'),  # supervised_keys is for lip reading only
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = os.path.join(dl_manager.manual_dir, 'lrs3_pretrain.zip')
        valid_path = os.path.join(dl_manager.manual_dir, 'lrs3_trainval.zip')
        test_path = os.path.join(dl_manager.manual_dir, 'lrs3_test_v0.4.zip')
        if not tf.io.gfile.exists(train_path) \
                or not tf.io.gfile.exists(valid_path) \
                or not tf.io.gfile.exists(test_path):
            raise AssertionError(
                'LRS3 requires manual download of the data. Please download '
                'the pretrain, trainval and test set and place them into: '
                '{}, {}, {}'.format(train_path, valid_path, test_path))
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'archive_path': train_path,
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'archive_path': valid_path,
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'archive_path': test_path,
                }
            ),
        ]

    def _generate_examples(self, archive_path):
        """Yields examples."""
        # this functions is adopted from
        # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/download/extractor.py
        def _normpath(path):
            path = os.path.normpath(path)
            if (path.startswith('.')
                    or os.path.isabs(path)
                    or path.endswith('~')
                    or os.path.basename(path).startswith('.')):
                return None
            return path.replace('\\', '/')  # For windows compatibility

        # yields examples without extract compressed archive files
        with tf.io.gfile.GFile(archive_path, 'rb') as fobj:
            z = zipfile.ZipFile(fobj)
            for member in z.infolist():
                # Filter directories and text files # pytype: disable=attribute-error
                if member.is_dir() or \
                    os.path.splitext(member.filename)[-1] != '.mp4':
                    continue
                video_path = _normpath(member.filename)
                if not video_path:
                    continue
                video_info = member
                video_fobj = z.open(video_info)
                video_bytes = video_fobj.read()

                # Caution: Video feature and audio feature should have separated
                #          file object. Since tfds.features.Video only can receive
                #          byte-stream. (which is a form of converted file object
                #          by 'read()') but tfds.features.Audio(using pydub) cannot.
                audio_fobj = z.open(video_info)

                text_path = os.path.splitext(video_path)[0] + '.txt'
                text_info = z.getinfo(text_path)
                text_fobj = z.open(text_info)
                text_bytes = text_fobj.readline()  # first line contains a sentence
                text_str = text_bytes.decode('utf-8')[len('Text:  '):-len('\n')]

                split, video_id, fname = video_path.split('/')
                record_id = '_'.join([video_id, os.path.splitext(fname)[0]])
                record = {
                    'file_name': video_path,
                    'video': video_bytes,
                    'audio': audio_fobj,
                    'sentence': text_str,
                    'video_id': video_id,
                }
                yield record_id, record
