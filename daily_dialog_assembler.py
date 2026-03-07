import json
from pathlib import Path


base_dir = Path("data/daily_dialog")

splits = {
    "train": base_dir / "train",
    "validation": base_dir / "validation",
    "test": base_dir / "test",
}

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_dialog_line(text_line):
    import re

    parts = text_line.split("__eou__")
    items = []

    parts = [x.strip() for x in parts if x.strip()]

    for x in parts:
        item = x.strip()
        if item:
            item = item.replace(" ' ", "'")
            item = item.replace(" ?", "?")
            item = item.replace(" )", ")")
            item = item.replace("( ", "(")
            item = item.replace(" !", "!")
            item = item.replace(" .", ".")
            item = item.replace(" ;", ";")
            item = item.replace(" ,", ",")
            item = item.replace(" ’ ", "'")
            item = item.replace(" / ", "/")
            item = item.replace(" & ", "&")
            item = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', item)
            items.append(item)
    return items


def parse_int_line(line):
    return [int(x) for x in line.strip().split() if x.strip()]


def build_split_dataset(split_dir, split_name):
    split_dir = Path(split_dir)

    text_path = split_dir / f"dialogues_{split_name}.txt"
    act_path = split_dir / f"dialogues_act_{split_name}.txt"
    emotion_path = split_dir / f"dialogues_emotion_{split_name}.txt"

    text_lines = read_lines(text_path)
    act_lines = read_lines(act_path)
    emotion_lines = read_lines(emotion_path)

    if not (len(text_lines) == len(act_lines) == len(emotion_lines)):
        raise ValueError(
            f"[{split_name}] The number of lines does not match: "
            f"text={len(text_lines)}, act={len(act_lines)}, emotion={len(emotion_lines)}"
        )

    items = []

    for i, (t, a, e) in enumerate(zip(text_lines, act_lines, emotion_lines), start=1):
        dialog = parse_dialog_line(t)
        acts = parse_int_line(a)
        emotions = parse_int_line(e)

        if not (len(dialog) == len(acts) == len(emotions)):
            raise ValueError(
                f"[{split_name}] Mismatch the number of utterances in the dialogue {i}: "
                f"dialog={len(dialog)}, act={len(acts)}, emotion={len(emotions)}"
            )

        items.append({
            "split": split_name,
            "dialog_id": i,
            "dialog": dialog,
            "act": acts,
            "emotion": emotions,
        })

    return items


def daily_dialog_expanded_gen_filter():

    file_path = "data/daily-dialog-expanded-gen/daily_dialog_expanded-all.txt"

    lines = []
    with open(Path(file_path), "r", encoding="utf-8") as fin:
        for line in fin:
            lines.append(line)

    #overwrrite the original file with filtered lines:
    with open(Path(file_path), "w", encoding="utf-8") as fout:
        for line in lines:
            line = line.replace(": , ", ": ")
            line = line.replace(". , ", ". ")
            line = line.replace(". . .", "...")
            line = line.replace(", ...", ",")
            line = line.replace(". .", ".")
            line = line.replace(", ,", ",")
            line = line.replace(", , ,", ",")
            line = line.replace(", , , ,", ",")
            fout.write(line)


def main():

    all_items = []

    for split_name, split_dir in splits.items():
        split_items = build_split_dataset(split_dir, split_name)
        all_items.extend(split_items)
        print(f"{split_name}: {len(split_items)} dialogs")

    output_path = base_dir / "daily-dialog_all.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(64*"-" + f"\nDone: {len(all_items)} dialogs \nSaved to: {output_path}")

    ################################################################################

    output_txt = base_dir / "daily-dialog_all.txt"

    with open(output_path, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            #fout.write(f"### SPLIT: {obj['split']} | DIALOGUE: {obj['dialog_id']}\n")
            for idx, utterance in enumerate(obj["dialog"], start=1):
                speaker = "Speaker1" if idx % 2 == 1 else "Speaker2"
                fout.write(f"{idx} {speaker}: {utterance}\n")
            fout.write("\n")

    print(f"Saved to: {output_txt}")

    daily_dialog_expanded_gen_filter()


if __name__ == "__main__":
    main()
