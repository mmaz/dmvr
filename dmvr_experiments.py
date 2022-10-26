# %%

from logging import captureWarnings
import matplotlib.pyplot as plt
import contextlib
import math
import os
from typing import Dict, Optional, Sequence
from pathlib import Path
import subprocess
import json

from absl import app
from absl import flags
import ffmpeg
import numpy as np
import pandas as pd
import tensorflow as tf

_JPEG_HEADER = b"\xff\xd8"

import examples.generate_from_file as gff

# %%
def get_audio_info(filepath: str):
    # ffprobe: does the video have audio, get duration, json oputput
    # https://stackoverflow.com/a/21447100
    # singlequotes to avoid issues with spaces in parentdir
    cmd = f"ffprobe -i '{filepath}' -show_streams -select_streams a -of json -loglevel error"
    # TODO(mmaz) json output and decoding might be slow
    out = subprocess.check_output(cmd, shell=True)
    audio_info = json.loads(out.decode("utf8"))
    # audio_info["name"] = name
    # audio_info["path"] = filepath
    # audio_info["category"] = category
    return audio_info


def has_audio(filepath: str):
    audio_info = get_audio_info(filepath)
    return len(audio_info["streams"]) > 0


# %%
def generate_sequence_example(
    video_path: str,
    start: float,
    end: float,
    label_name: Optional[str] = None,
    caption: Optional[str] = None,
    label_map: Optional[Dict[str, int]] = None,
):
    """Generate a sequence example."""
    # TODO(mmaz) this is probably slow
    if not has_audio(video_path):
        raise ValueError("video has no audio")
    #   if FLAGS.video_root_path:
    #     video_path = os.path.join(FLAGS.video_root_path, video_path)
    imgs_encoded = gff.extract_frames(video_path, start, end)

    # Initiate the sequence example.
    seq_example = tf.train.SequenceExample()

    # Add the label list as text and indices.
    if label_name:
        gff.set_context_int("clip/label/index", label_map[label_name], seq_example)
        gff.set_context_bytes("clip/label/text", label_name.encode(), seq_example)
    if caption:
        gff.set_context_bytes("caption/string", caption.encode(), seq_example)
    # Add the frames as one feature per frame.
    for img_encoded in imgs_encoded:
        gff.add_bytes_list("image/encoded", [img_encoded], seq_example)

    # Add audio.
    audio = gff.extract_audio(video_path, start, end)
    gff.add_float_list("WAVEFORM/feature/floats", audio, seq_example)

    # Add other metadata.
    gff.set_context_bytes("video/filename", video_path.encode(), seq_example)
    # Add start and time in micro seconds.
    gff.set_context_int("clip/start/timestamp", int(1000000 * start), seq_example)
    gff.set_context_int("clip/end/timestamp", int(1000000 * end), seq_example)
    return seq_example


# %%
record_fp = Path.home() / "apple/tmp/test.tfrecord"
record_fp = str(record_fp)

# %%
k700 = Path("/media/mark/sol/kinetics-dataset/k700-2020")
ktrain = k700 / "train"
vid_dir = ktrain / "dribbling basketball"
vids = list(vid_dir.glob("*.mp4"))

# https://stackoverflow.com/a/31025482
def get_length(input_video):
    # fmt: off
    result = subprocess.run(
        [ "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_video, ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # fmt: on
    return float(result.stdout)


# %%
vid = str(vids[5])
print(vid)
end = get_length(vid)
print("length", end)
seq_ex = generate_sequence_example(vid, start=0, end=end)

# %%
# read both of these:
# https://www.tensorflow.org/api_docs/python/tf/train/FloatList
# https://www.tensorflow.org/api_docs/python/tf/io/parse_sequence_example
# https://trac.ffmpeg.org/wiki/audio%20types
data = (
    seq_ex.feature_lists.feature_list["WAVEFORM/feature/floats"]
    .feature[0]
    .float_list.value
)
print(len(data))

rec = tf.io.parse_sequence_example(
    seq_ex.SerializeToString(),
    sequence_features={
        "WAVEFORM/feature/floats": tf.io.RaggedFeature(dtype=tf.float32)
    },
)

print(rec[1].keys())
audio = rec[1]["WAVEFORM/feature/floats"].numpy()
print(audio.shape)

# %%
plt.plot(np.squeeze(audio))

# %%
audio[0, 200:240]
# %%

# %%
# write the record back out
reencoded = tf.audio.encode_wav(audio.T, 16000)
tf.io.write_file("test.wav", reencoded)

# %%
writer = tf.io.TFRecordWriter(record_fp)
writer.close()
# %%
vid
# %%
# end to end encoding example
sampling_rate = 16_000
# vid = "/media/mark/sol/kinetics-dataset/k700-2020/train/dribbling basketball/D7URTg7KuMw_000008_000018.mp4"
cmd = ffmpeg.input(vid).output("pipe:", ac=1, ar=sampling_rate, format="s16le")
audio, _ = cmd.run(capture_stdout=True, quiet=True)
# ffmpeg returns int16 encoded bytes
audio = np.frombuffer(audio, np.int16)
print("audio shape as int16", audio.shape)
plt.plot(audio)
# normalize to [-1, 1]
audio_f32 = audio.astype(np.float32) / 32768.0  # https://stackoverflow.com/a/42544738
# expand to length, num_channels
reencoded = tf.audio.encode_wav(np.expand_dims(audio_f32, -1), sampling_rate)
tf.io.write_file("test.wav", reencoded)

# reencoded = tf.audio.encode_wav(audio.T, 16000)
# tf.io.write_file("test.wav", reencoded)
# %%
vid
# %%
# dmvr/modalities.py : add_image()
rec = tf.io.parse_sequence_example(
    seq_ex.SerializeToString(),
    sequence_features={
        "image/encoded": tf.io.FixedLenSequenceFeature((), dtype=tf.string)
    },
)
print(rec[1].keys())

# tf warning:
# back_prop=False is deprecated. Consider using tf.stop_gradient instead.
# Instead of:
# results = tf.map_fn(fn, elems, back_prop=False)
# Use:
# results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))

# channels=0 means channels are calculated at runtime

# dmvr/processors.py: decode_jpeg()
image_string = rec[1]["image/encoded"].numpy()
images = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(
    lambda x: tf.image.decode_jpeg(x, channels=0),
    image_string,
    dtype=tf.uint8,
))
print(images.shape)

# %%
plt.imshow(images[0])

# %%
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
        raise ValueError(
            "Spectrogram type should be one of `spectrogram`, "
            "`logmf`, or `mfcc`, got {}".format(spectrogram_type)
        )
    if normalize:
        raw_audio /= tf.reduce_max(tf.abs(raw_audio), axis=-1, keepdims=True) + 1e-8
        # features[audio_feature_name] = raw_audio
    #   if preemphasis is not None:
    #     raw_audio = _preemphasis(raw_audio, preemphasis)

    def _extract_spectrogram(waveform: tf.Tensor, spectrogram_type: str) -> tf.Tensor:
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
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return log_mel_spectrograms

    spectrogram = _extract_spectrogram(raw_audio, spectrogram_type)
    return spectrogram


specs = to_spec(audio_f32)
print(specs.shape)
# %%
# %%
plt.imshow(specs.numpy().T)

# %%
