import os
# hide gpu from TF so we can parallelize spectrogram generation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dataclasses import dataclass
import datetime
import logging
import matplotlib.pyplot as plt
import contextlib
import math
import os
import sys
from typing import Dict, Optional, Sequence
from pathlib import Path
import subprocess
import multiprocessing
import concurrent.futures
import time
import json

from absl import app
from absl import flags
import ffmpeg
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.experimental.numpy as tnp
import tqdm

_JPEG_HEADER = b"\xff\xd8"

src = str(Path.home() / "apple/mmaz_dmr/")
sys.path.append(src)
import examples.generate_from_file as gff


def to_spec(
    raw_audio,
    spectrogram_type="logmf",
    sample_rate=16_000,
    frame_length: int = 400,
    frame_step: int = 160,
    num_features: int = 128,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    normalize: bool = False,
):
    """
    this is modified from dmvr/processors.py: compute_audio_spectrogram()

    frame_length and frame_step are in passed in as number of samples
    so for 16kHz, 25ms is 400 samples (for frame length)
    16_000 samples per sec * 0.025 sec = 400 samples
    frame_step is 10ms, so 160 samples
    """
    if spectrogram_type not in ["spectrogram", "logmf", "mfcc"]:
        raise ValueError("Spectrogram type should be one of `spectrogram`, "
                         "`logmf`, or `mfcc`, got {}".format(spectrogram_type))
    if normalize:
        raw_audio /= tf.reduce_max(tf.abs(raw_audio), axis=-1,
                                   keepdims=True) + 1e-8
        # features[audio_feature_name] = raw_audio
    #   if preemphasis is not None:
    #     raw_audio = _preemphasis(raw_audio, preemphasis)

    def _extract_spectrogram(waveform: tf.Tensor,
                             spectrogram_type: str) -> tf.Tensor:
        # NOTE(mmaz): modified to use a hamming_window instead of tf.signal.hann
        stfts = tf.signal.stft(
            waveform,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=frame_length,
            window_fn=tf.signal.hamming_window,
            pad_end=True,
        )
        spectrograms = tf.abs(stfts)

        if spectrogram_type == "spectrogram":
            return spectrograms[..., :num_features]

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_features,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrograms,
                                        linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return log_mel_spectrograms

    spectrogram = _extract_spectrogram(raw_audio, spectrogram_type)
    return spectrogram


def generate_sequence_example(
        video_path: str,
        label_id: int,
        label_name: str,
        start: float,
        end: float,
        fps: int,
        min_resize: int,
        sampling_rate: int,
        spec_width_timeaxis: int = 100,  # 100x128
) -> Optional[tf.train.SequenceExample]:
    """Generate a sequence example."""
    imgs_encoded = gff.extract_frames(video_path,
                                      start,
                                      end,
                                      fps=fps,
                                      min_resize=min_resize)

    # Initiate the sequence example.
    seq_example = tf.train.SequenceExample()

    # Add the frames as one feature per frame.
    for img_encoded in imgs_encoded:
        gff.add_bytes_list("image/encoded", [img_encoded], seq_example)

    # Add audio.
    audio = gff.extract_audio(video_path,
                              start,
                              end,
                              sampling_rate=sampling_rate)

    spec = to_spec(audio, spectrogram_type="logmf", sample_rate=sampling_rate)
    nearest_divisible = spec.shape[0] - (spec.shape[0] % spec_width_timeaxis)
    spec_divisible = spec[:nearest_divisible, :]

    for spec_second in np.vsplit(spec_divisible.numpy(),
                                 nearest_divisible // spec_width_timeaxis):
        gff.add_float_list("melspec/feature/floats",
                           np.reshape(spec_second, -1), seq_example)

    # to include raw waveforms:
    # gff.add_float_list("WAVEFORM/feature/floats", audio, seq_example)

    # Add other metadata.
    gff.set_context_bytes("video/filename", video_path.encode(), seq_example)
    # Add start and time in micro seconds.
    gff.set_context_int("clip/start/timestamp", int(1_000_000 * start),
                        seq_example)
    gff.set_context_int("clip/end/timestamp", int(1_000_000 * end), seq_example)

    # see dmvr/modalities.py::add_label()
    # > This function expects the input to be either a `tf.train.SequenceExample`
    # > (with the features in the context) or a `tf.train.Example`. The expected
    # > structure is (or equivalent for `tf.train.Example`):

    # the inputs to this function do not support multilabel classification at the moment
    gff.set_context_int_list("clip/label/index", [label_id], seq_example)
    gff.set_context_bytes("clip/label/text", label_name.encode(), seq_example)

    # if you have multiple label_ids (multitarget classification):
    # label = ",".join([f"label{x}" for x in random_labels])
    # gff.set_context_bytes("clip/label/text", label.encode(), seq_example)

    return seq_example


@dataclass
class VidToEncode:
    ix: int
    video_path: str
    duration: float
    category: str
    label_id: int


def create_sequence(target: VidToEncode):
    seq_ex = generate_sequence_example(target.video_path,
                                       start=0,
                                       end=target.duration,
                                       label_id=target.label_id,
                                       label_name=target.category,
                                       fps=25,
                                       min_resize=256,
                                       sampling_rate=16_000)
    return seq_ex, target


def write_split():
    # load kinetics subset with audio (all videos > 8s in duration)
    kinetics_subset_waudio = json.loads(
        Path("kinetics_subset_waudio_duration_over8sec.json").read_text())

    split = "train"
    print(f"{split=}")
    basedir = Path("/media/mark/sol/kinetics_sound/")
    split_dir = str(basedir / split)
    assert Path(split_dir).exists(), f"{split_dir} does not exist"
    assert len(list(Path(split_dir).iterdir())) == 0, f"{split_dir} not empty"
    basename = "kinetics_sound"

    categories = list(sorted(kinetics_subset_waudio[split].keys()))
    category2labelid = {c: ix for ix, c in enumerate(categories)}
    print("num categories", len(categories))
    all_videos = [
        v for c in categories for v in kinetics_subset_waudio[split][c]
    ]
    print("num vids", len(all_videos), "avg",
          len(all_videos) // len(categories))
    num_shards = int(math.sqrt(len(all_videos)))
    print(f"{num_shards=}")

    # add enumeration to kinetics_subset_waudio
    # this enables indexing into the list of writers
    encoding_targets = []
    idx = 0
    # categories are already sorted by lexical order in the json, but double checking
    for category, videos in sorted(kinetics_subset_waudio[split].items(),
                                   key=lambda x: x[0]):
        for video in sorted(videos, key=lambda x: x["path"]):
            video_path = video["path"]
            duration = video["duration"]
            target = VidToEncode(ix=idx,
                                 video_path=video_path,
                                 duration=duration,
                                 category=category,
                                 label_id=category2labelid[category])
            encoding_targets.append(target)
            idx += 1
    print("num vids to encode", len(encoding_targets))
    print(encoding_targets[0])

    video_filesizes = [os.path.getsize(v.video_path) for v in encoding_targets]
    print("videos in GB", sum(video_filesizes) / (1024**3))

    rng = np.random.default_rng(0)
    shuffled_targets = rng.permutation(encoding_targets)

    print("writing shards")
    total_to_write = len(shuffled_targets)

    shard_names = [
        os.path.join(split_dir, f"{basename}-{i:05d}-of-{num_shards:05d}")
        for i in range(num_shards)
    ]
    writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]

    written = 0
    start_time = datetime.datetime.now()
    with gff._close_on_exit(writers) as writers:
        with multiprocessing.Pool() as pool:
            for seq_ex, target in pool.imap_unordered(create_sequence,
                                                      shuffled_targets,
                                                      chunksize=50):
                writer = writers[target.ix % num_shards]
                writer.write(seq_ex.SerializeToString())
                written += 1
                if target.ix % 250 == 0:
                    current = datetime.datetime.now()
                    pct_written = written / total_to_write
                    print(
                        f"{target.ix=}, {written=}, {pct_written:0.2f}, timedelta={current - start_time}"
                    )
    print(f"total {written=}")


if __name__ == "__main__":
    write_split()