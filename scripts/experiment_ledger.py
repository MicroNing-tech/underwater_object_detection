
from pathlib import Path
import csv


# ==========================================
# 1. 实验台账字段
# ==========================================

FIELDS = [
    # 实验信息
    "experiment_name",
    "model",
    "dataset",

    # 训练参数
    "epochs",
    "actual_epochs",
    "best_epoch",
    "batch",
    "imgsz",
    "optimizer",
    "seed",

    # 模型复杂度
    "parameters_M",
    "gflops",

    # 硬件资源
    "gpu_name",
    "gpu_peak_memory_gib",

    # 验证集指标
    "val_precision",
    "val_recall",
    "val_mAP50",
    "val_mAP50_95",

    # 测试集指标
    "test_precision",
    "test_recall",
    "test_mAP50",
    "test_mAP50_95",

    # 训练耗时
    "train_minutes",
]


# ==========================================
# 2. 保存实验记录
# ==========================================

def save_experiment_record(record, project_root):

    save_dir = Path(project_root) / "records"
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "experiment_ledger.csv"

    is_new = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    # 检查已有台账表头
    if not is_new:
        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            header = next(csv.reader(f), [])

        if header != FIELDS:
            raise ValueError(
                "现有台账字段不一致，请先备份旧 CSV。"
            )

    # 追加一条实验记录
    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8-sig" if is_new else "utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        if is_new:
            writer.writeheader()

        writer.writerow({
            field: record.get(field, "")
            for field in FIELDS
        })

    print(f"\n实验台账已保存: {csv_path}")
