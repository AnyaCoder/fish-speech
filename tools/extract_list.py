import argparse
import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 114514
# lang_dict = {'en': 'EN', 'zh-cn': "ZH", 'zh-tw': 'ZH', 'ja': "JP", 'ko': "KO"}
import langid
from collections import defaultdict
lang_dict = {'en': 'EN', 'zh': 'ZH', 'ja': "JP"}
langid.set_languages(['en', 'zh', 'ja'])
project_root = str(Path(__file__).parent.parent.resolve())

def extract_list(folder_path, transcript_file):

    folder_path = Path(folder_path)
    transcript_file = Path(transcript_file)

    transcript_file.parent.mkdir(parents=True, exist_ok=True)

    with transcript_file.open("w", encoding="utf-8") as f:
        k_lang_v_info = defaultdict(lambda: defaultdict(list))
        k_char_v_lang = defaultdict(str)
        for lab_path in tqdm(folder_path.rglob("*.lab")):
            transcription = lab_path.read_text(encoding="utf-8").strip()
            if len(transcription) == 0:
                continue

            wav_path = lab_path.with_suffix(".wav")
            relative_path = wav_path.relative_to(folder_path)
            parts = relative_path.parts
            if parts:
                first_folder_name = parts[0]
            else:
                first_folder_name = "unknown"
            try:
                language = lang_dict[langid.classify((first_folder_name + ',' + transcription) * 2)[0]]
            except:
                language = "ZH"

            k_lang_v_info[first_folder_name][language].append(dict(
                wav_path=str(wav_path.relative_to(project_root)),
                transcription=transcription
            ))
        # langid is not 100% accurate! so the following is for robustness.
        for character, ex_C_info in k_lang_v_info.items():
            max_len = 0
            max_lang = "ZH"
            for lang, lst in ex_C_info.items():
                if len(lst) > max_len:
                    max_len = len(lst)
                    max_lang = lang
            for lang, lst in ex_C_info.items():
                for info in lst:
                    wav_path = Path(info["wav_path"]).resolve()
                    transcription = info["transcription"]
                    if wav_path.is_file():
                        line = f"{wav_path.relative_to(project_root)}|{character}|{max_lang}|{transcription}\n"
                        f.write(line)
                    else:
                        print(f"No such file or directory: {wav_path}")

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
