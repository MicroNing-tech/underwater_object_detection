from pathlib import Path

import yaml


# ==========================================
# 1. 配置路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    r"E:\learn_postgraduate\Dataset\RUOD-R\RUOD-R-YOLO-Split"
)

DATA_YAML = PROJECT_ROOT / "configs" / "dataset" / "data.yaml"

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

SPLITS = ("train", "val", "test")


# ==========================================
# 2. 检查标签格式
# ==========================================

def check_label_file(label_path, num_classes):
    """检查 YOLO 检测标签是否符合标准格式。"""

    errors = []

    lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_no, line in enumerate(lines, start=1):

        if not line.strip():
            continue

        parts = line.split()

        # YOLO 检测标签：
        # class_id x_center y_center width height
        if len(parts) != 5:
            errors.append(
                f"{label_path} 第 {line_no} 行："
                "标注字段数量不是 5"
            )
            continue

        try:
            values = [float(x) for x in parts]
        except ValueError:
            errors.append(
                f"{label_path} 第 {line_no} 行："
                "存在非数字数据"
            )
            continue

        class_id, x, y, w, h = values

        # 类别编号检查
        if (
            not class_id.is_integer()
            or not 0 <= class_id < num_classes
        ):
            errors.append(
                f"{label_path} 第 {line_no} 行："
                "类别编号不合法"
            )

        # 归一化坐标检查
        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            errors.append(
                f"{label_path} 第 {line_no} 行："
                "归一化坐标不合法"
            )

    return errors


# ==========================================
# 3. 检查数据集
# ==========================================

def check_dataset():

    print("\n========== 数据集检查 ==========")

    # 检查 YAML 文件
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"YAML 文件不存在: {DATA_YAML}"
        )

    # 检查数据集目录
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"数据集目录不存在: {DATA_ROOT}"
        )

    # 读取 YAML（仅读取，不修改）
    with DATA_YAML.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML 文件格式不正确")

    names = config.get("names")

    if not isinstance(names, (list, dict)) or not names:
        raise ValueError("YAML 缺少有效的 names 配置")

    num_classes = len(names)

    if (
        "nc" in config
        and int(config["nc"]) != num_classes
    ):
        raise ValueError("YAML 中 nc 与 names 数量不一致")

    print(f"类别数量: {num_classes}")

    all_errors = []

    # 检查 train、val、test
    for split in SPLITS:

        if not config.get(split):
            raise ValueError(
                f"YAML 缺少 {split} 路径配置"
            )

        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"图片目录不存在: {image_dir}"
            )

        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"标签目录不存在: {label_dir}"
            )

        # 获取所有图片
        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
        )

        # 获取所有标签
        labels = sorted(label_dir.rglob("*.txt"))

        if not images:
            raise ValueError(
                f"{split} 数据集没有图片"
            )

        if not labels:
            raise ValueError(
                f"{split} 数据集没有标签文件"
            )

        # 检查图片与标签的对应关系
        image_keys = {
            p.relative_to(image_dir).with_suffix("")
            for p in images
        }

        label_keys = {
            p.relative_to(label_dir).with_suffix("")
            for p in labels
        }

        missing_labels = image_keys - label_keys
        extra_labels = label_keys - image_keys

        # 检查每个标签文件的内容
        for label_path in labels:
            errors = check_label_file(
                label_path,
                num_classes
            )
            all_errors.extend(errors)

        print(f"\n[{split}]")
        print(f"图片数量: {len(images)}")
        print(f"标签文件数量: {len(labels)}")
        print(f"无对应标签的图片: {len(missing_labels)}")
        print(f"无对应图片的标签: {len(extra_labels)}")

        if missing_labels:
            print(
                "提示：没有标签文件的图片可能是"
                "合法的背景图片，需要确认。"
            )

        if extra_labels:
            print("警告：存在没有对应图片的标签文件。")

    # 输出标签格式检查结果
    print("\n========== 检查结果 ==========")

    if all_errors:
        print(f"发现 {len(all_errors)} 处标签格式问题")

        for error in all_errors[:10]:
            print(error)

        raise ValueError("数据集存在标签格式问题")

    print("标签格式检查通过！")
    print("数据集目录检查完成！")


# ==========================================
# 4. 程序入口
# ==========================================

if __name__ == "__main__":
    check_dataset()
