from pathlib import Path
import json
import shutil
import tqdm


def main():
    destdir = Path.home() / "apple/kinetics-sound-source"
    samples = json.loads(
        (destdir / "kinetics_subset_waudio_duration_over8sec.json").read_text()
    )
    for split in ["train", "val"]:
        for category, videos in tqdm.tqdm(samples[split].items()):
            for video in videos:
                src_path = Path(video["path"])
                dest_path = destdir / split / category
                # print(src_path)
                # print(dest_path)
                dest_path.mkdir(exist_ok=True)
                shutil.copy2(src_path, dest_path)



if __name__ == "__main__":
    main()
