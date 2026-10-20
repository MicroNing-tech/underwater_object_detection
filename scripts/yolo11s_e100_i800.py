from pathlib import Path
import random

import torch
from ultralytics import YOLO


# ==========================================
# 1. 配置路径
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = Path(
    r"E:\learn_postgraduate\Dataset\RUOD-R\RUOD-R-YOLO-Split"
)

DATA_YAML = PROJECT_ROOT / "configs" / "dataset" / "data.yaml"

PROJECT_DIR = PROJECT_ROOT / "runs"

EXPERIMENT_NAME = "yolo11s_epoch100_imgsz800"


# ==========================================
# 2. 配置训练参数
# ==========================================

EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640

DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 0
SEED = 42
PATIENCE = 10

SAVE_SAMPLE_PREDICTIONS = True
SAMPLE_PREDICTION_COUNT = 20
PREDICTION_CONF = 0.25

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


# ==========================================
# 3. 打印评估指标
# ==========================================

def print_metrics(metrics, split_name):

    print(f"\n====== {split_name} 评估结果 ======")

    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")


# ==========================================
# 4. 保存预测示例
# ==========================================

def save_sample_predictions(model, run_dir):

    if not SAVE_SAMPLE_PREDICTIONS:
        return

    print("\n========== 保存预测示例 ==========")

    # 测试集图片目录
    test_dir = DATA_ROOT / "images" / "test"

    # 获取测试集图片
    test_images = sorted(
        p for p in test_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if not test_images:
        print("测试集没有可供预测的图片。")
        return

    # 随机选取最多 20 张测试图片
    sample_count = min(
        SAMPLE_PREDICTION_COUNT,
        len(test_images)
    )

    rng = random.Random(SEED)

    sample_images = rng.sample(
        test_images,
        sample_count
    )

    # 创建预测保存目录
    save_dir = run_dir / "sample_predictions"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型预测
    results = model.predict(
        source=[str(p) for p in sample_images],
        imgsz=IMG_SIZE,
        conf=PREDICTION_CONF,
        device=DEVICE,
        stream=True,
        save=False,
        verbose=False,
    )

    # 保存带检测框的图片
    for index, result in enumerate(results, start=1):

        image_name = Path(result.path).stem

        save_path = save_dir / (
            f"{index:02d}_{image_name}.jpg"
        )

        result.save(filename=str(save_path))

        print(
            f"[{index}/{sample_count}] "
            f"已保存: {save_path.name}"
        )

    print("预测示例保存完成！")


# ==========================================
# 5. 主函数
# ==========================================

def main():

    # --------------------------------------
    # 5.1 打印训练设备
    # --------------------------------------

    print(f"当前训练设备: {DEVICE}")

    if torch.cuda.is_available():
        print(
            "GPU 名称:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------
    # 5.2 加载预训练模型
    # --------------------------------------

    model = YOLO("yolo11s.pt")
    model.info()

    # --------------------------------------
    # 5.3 模型训练
    # --------------------------------------

    print("\n========== 开始训练 ==========")

    model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,

        device=DEVICE,
        workers=WORKERS,

        seed=SEED,
        deterministic=True,

        patience=PATIENCE,
        optimizer="auto",

        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )

    print("\n训练完成！")

    # --------------------------------------
    # 5.4 加载最佳模型
    # --------------------------------------

    best_path = Path(model.trainer.best)

    if not best_path.is_file():
        raise FileNotFoundError(
            f"最佳权重不存在: {best_path}"
        )

    run_dir = Path(model.trainer.save_dir)

    print(f"最佳模型路径: {best_path}")

    best_model = YOLO(str(best_path))

    # --------------------------------------
    # 5.5 验证集评估
    # --------------------------------------

    print("\n========== 验证集评估 ==========")

    val_metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",

        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        plots=True,

        project=str(run_dir),
        name="validation",
        exist_ok=False,
    )

    print_metrics(val_metrics, "验证集")

    # --------------------------------------
    # 5.6 测试集评估
    # --------------------------------------

    print("\n========== 测试集评估 ==========")

    test_metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",

        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        plots=True,

        project=str(run_dir),
        name="testing",
        exist_ok=False,
    )

    print_metrics(test_metrics, "测试集")

    # --------------------------------------
    # 5.7 保存预测示例
    # --------------------------------------

    save_sample_predictions(
        model=best_model,
        run_dir=run_dir
    )

    print("\n====================================")
    print("YOLO11s 训练、验证、测试全部完成！")
    print("====================================")

    print(f"最佳模型: {best_path}")
    print(f"结果目录: {run_dir}")


# ==========================================
# 6. 程序入口
# ==========================================

if __name__ == "__main__":
    main()
