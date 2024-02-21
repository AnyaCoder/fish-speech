import argparse
import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 114514
# lang_dict = {'en': 'EN', 'zh-cn': "ZH", 'zh-tw': 'ZH', 'ja': "JP", 'ko': "KO"}
import langid
lang_dict = {'en': 'EN', 'zh': 'ZH', 'ja': "JP"}
langid.set_languages(['en', 'zh', 'ja'])
project_root = str(Path(__file__).parent.parent.resolve())

def extract_list(folder_path, transcript_file):

    folder_path = Path(folder_path)
    transcript_file = Path(transcript_file)

    transcript_file.parent.mkdir(parents=True, exist_ok=True)

    with transcript_file.open("w", encoding="utf-8") as f:
        for lab_file_path in tqdm(folder_path.rglob("*.lab")):
            transcription = lab_file_path.read_text(encoding="utf-8").strip()
            if len(transcription) == 0:
                continue

            wav_file_path = lab_file_path.with_suffix(".wav")
            relative_path = wav_file_path.relative_to(folder_path)
            parts = relative_path.parts
            if parts:
                first_folder_name = parts[0]
            else:
                first_folder_name = "unknown"
            try:
                language = lang_dict[langid.classify((first_folder_name + ',' + transcription) * 2)[0]]
            except:
                language = "ZH"
            if wav_file_path.is_file():
                line = f"{str(wav_file_path.relative_to(project_root))}|{first_folder_name}|{language}|{transcription}\n"
                f.write(line)
            else:
                logger.warning(f"不存在对应音频 {wav_file_path}!")

    return f"转写文本 {transcript_file} 生成完成"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default=str(Path("data/demo").resolve()),
                        help="path of your rawaudios, e.g. ./Data/xxx/audios/raw")
    parser.add_argument("-o", "--outfile", default=str(Path("data/demo/detect.list").resolve()),
                        help="output transcript listfile(containing all .lab)")
    args = parser.parse_args()

    status_str = extract_list(args.folder, args.outfile)
    logger.critical(status_str)
