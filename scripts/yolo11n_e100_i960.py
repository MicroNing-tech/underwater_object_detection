from pathlib import Path
import random
import time

import torch
from ultralytics import YOLO

from experiment_ledger import save_experiment_record
from inference_benchmark import benchmark_fps


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

EXPERIMENT_NAME = "yolo11n_epoch100_imgsz960"

MODEL_NAME = "YOLO11n"

PRETRAINED_WEIGHTS = "yolo11n.pt"

DATASET_NAME = "RUOD-R"


# ==========================================
# 2. 配置训练参数
# ==========================================

EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 960

DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 0
SEED = 42
PATIENCE = 10
OPTIMIZER = "auto"

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
        return 0

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
        return 0

    # 随机选取测试图片
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

    saved_count = 0

    # 保存带检测框的图片
    for index, result in enumerate(results, start=1):

        image_name = Path(result.path).stem

        save_path = save_dir / (
            f"{index:02d}_{image_name}.jpg"
        )

        result.save(filename=str(save_path))

        saved_count += 1

        print(
            f"[{index}/{sample_count}] "
            f"已保存: {save_path.name}"
        )

    print(f"预测示例保存完成，共{saved_count}张。")

    return saved_count


# ==========================================
# 5. 主函数
# ==========================================

def main():

    # --------------------------------------
    # 5.1 打印训练设备
    # --------------------------------------

    print("\n========== 训练环境 ==========")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(DEVICE)
    else:
        gpu_name = "CPU"

    print(f"训练设备: {DEVICE}")
    print(f"设备名称: {gpu_name}")

    # --------------------------------------
    # 5.2 加载预训练模型
    # --------------------------------------

    print("\n========== 加载模型 ==========")

    model = YOLO(PRETRAINED_WEIGHTS)
    model.info()

    # --------------------------------------
    # 5.3 模型训练
    # --------------------------------------

    print("\n========== 开始训练 ==========")

    # 重置 GPU 显存峰值统计
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # 开始计时
    train_start = time.perf_counter()

    # 执行训练
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
        optimizer=OPTIMIZER,

        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )

    # 确保 GPU 操作完成
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)

    # 训练耗时（分钟）
    train_minutes = (
        time.perf_counter() - train_start
    ) / 60

    # --------------------------------------
    # 5.4 获取 GPU 峰值显存
    # --------------------------------------

    if torch.cuda.is_available():

        gpu_peak_memory_gib = (
            torch.cuda.max_memory_reserved(DEVICE)
            / (1024 ** 3)
        )

    else:
        gpu_peak_memory_gib = None

    print("\n========== 训练资源统计 ==========")

    print(f"训练耗时: {train_minutes:.2f} min")

    if gpu_peak_memory_gib is not None:
        print(
            f"训练峰值显存: "
            f"{gpu_peak_memory_gib:.2f} GiB"
        )

    # --------------------------------------
    # 5.5 获取训练信息
    # --------------------------------------

    trainer = model.trainer

    # 实际完成的训练轮数
    actual_epochs = int(trainer.epoch) + 1

    # 最佳 Epoch
    stopper = getattr(trainer, "stopper", None)

    best_epoch = getattr(
        stopper, "best_epoch", None
    )

    if best_epoch is not None and best_epoch > 0:
        best_epoch = int(best_epoch)
    else:
        best_epoch = ""

    # 获取实际优化器
    optimizer_object = getattr(
        trainer, "optimizer", None
    )

    if optimizer_object is not None:
        optimizer_name = type(
            optimizer_object
        ).__name__
    else:
        optimizer_name = OPTIMIZER

    print("\n========== 训练轮次信息 ==========")

    print(f"设定 Epoch: {EPOCHS}")
    print(f"实际 Epoch: {actual_epochs}")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"实际优化器: {optimizer_name}")

    # --------------------------------------
    # 5.6 加载最佳模型
    # --------------------------------------

    print("\n========== 加载最佳模型 ==========")

    best_path = Path(trainer.best).resolve()

    if not best_path.is_file():
        raise FileNotFoundError(
            f"最佳模型权重不存在: {best_path}"
        )

    run_dir = Path(trainer.save_dir).resolve()

    print(f"最佳模型路径: {best_path}")

    best_model = YOLO(str(best_path))

    # --------------------------------------
    # 5.7 统计模型复杂度
    # --------------------------------------

    print("\n========== 模型复杂度 ==========")

    # 参数量（百万）
    parameters_M = (
            sum(
                p.numel()
                for p in best_model.model.parameters()
            ) / 1e6
    )

    # 计算 GFLOPs
    # 注意：verbose=True 才返回模型统计信息
    model_info = best_model.info(
        imgsz=IMG_SIZE,
        verbose=True
    )

    if (
            model_info is not None
            and len(model_info) >= 4
    ):
        gflops = model_info[3]
    else:
        gflops = None

    # GFLOPs 为 0 可能表示分析失败
    if gflops is not None and gflops > 0:
        gflops = round(float(gflops), 2)
    else:
        gflops = ""

    print(f"模型参数量: {parameters_M:.3f} M")
    print(f"GFLOPs: {gflops}")

    # --------------------------------------
    # 5.8 验证集评估
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
    # 5.9 测试集评估
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
    # 5.10 保存预测示例
    # --------------------------------------

    save_sample_predictions(
        model=best_model,
        run_dir=run_dir
    )

    # ==========================================
    # 5.11 测量端到端 FPS
    # ==========================================

    fps_mean, fps_std = benchmark_fps(
        model=best_model,
        test_dir=DATA_ROOT / "images" / "test",
        imgsz=IMG_SIZE,
        device=DEVICE,
        batch=1,
        warmup=50,
        repeats=3,
    )

    # --------------------------------------
    # 5.12 整理实验台账
    # --------------------------------------

    print("\n========== 整理实验台账 ==========")

    record = {

        # ===== 实验信息 =====
        "experiment_name": run_dir.name,
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,

        # ===== 训练参数 =====
        "epochs": EPOCHS,
        "actual_epochs": actual_epochs,
        "best_epoch": best_epoch,

        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "optimizer": optimizer_name,
        "seed": SEED,

        # ===== 模型复杂度 =====
        "parameters_M": round(parameters_M, 3),
        "gflops": gflops,

        # ===== 硬件资源 =====
        "gpu_name": gpu_name,

        "gpu_peak_memory_gib": (
            round(gpu_peak_memory_gib, 2)
            if gpu_peak_memory_gib is not None
            else ""
        ),

        # ===== 验证集指标 =====
        "val_precision": round(
            float(val_metrics.box.mp), 4
        ),

        "val_recall": round(
            float(val_metrics.box.mr), 4
        ),

        "val_mAP50": round(
            float(val_metrics.box.map50), 4
        ),

        "val_mAP50_95": round(
            float(val_metrics.box.map), 4
        ),

        # ===== 测试集指标 =====
        "test_precision": round(
            float(test_metrics.box.mp), 4
        ),

        "test_recall": round(
            float(test_metrics.box.mr), 4
        ),

        "test_mAP50": round(
            float(test_metrics.box.map50), 4
        ),

        "test_mAP50_95": round(
            float(test_metrics.box.map), 4
        ),

        # ===== 训练时间 =====
        "train_minutes": round(
            train_minutes, 2
        ),

        # ===== 推理 FPS =====
        "fps_mean": round(
            fps_mean, 3
        ),

        "fps_std": round(
            fps_std, 3
        ),
    }

    # --------------------------------------
    # 5.13 调用台账记录函数
    # --------------------------------------

    save_experiment_record(
        record=record,
        project_root=PROJECT_ROOT
    )

    # --------------------------------------
    # 5.14 运行完成
    # --------------------------------------

    print("\n====================================")
    print("YOLO11n 实验全部完成！")
    print("====================================")

    print(f"最佳模型: {best_path}")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"模型参数量: {parameters_M:.3f} M")
    print(f"GFLOPs: {gflops}")
    print(f"训练耗时: {train_minutes:.2f} min")

    print(f"实验目录: {run_dir}")


# ==========================================
# 6. 程序入口
# ==========================================

if __name__ == "__main__":
    main()
