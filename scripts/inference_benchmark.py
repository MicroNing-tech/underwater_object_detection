from pathlib import Path
import time
import statistics

import torch


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def _sync(device):
    """等待 GPU 计算完成，保证计时准确。"""

    if not torch.cuda.is_available():
        return

    if str(device).lower() == "cpu":
        return

    if isinstance(device, int) or str(device).isdigit():
        torch.cuda.synchronize(int(device))
    else:
        torch.cuda.synchronize(device)


def benchmark_fps(
    model,
    test_dir,
    imgsz=640,
    device=0,
    batch=1,
    warmup=50,
    repeats=3,
):
    """
    测量完整测试集的端到端 FPS。

    包含：图片读取、解码、预处理、模型推理、后处理。
    不包含：模型加载、预热、结果可视化和文件保存。

    返回：
        inference_fps: FPS 均值
        inference_fps_std: 多次测试的标准差
    """

    test_dir = Path(test_dir)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"测试集目录不存在: {test_dir}"
        )

    image_paths = sorted(
        p for p in test_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if not image_paths:
        raise RuntimeError(
            f"测试集没有图片: {test_dir}"
        )

    if batch < 1 or warmup < 1 or repeats < 1:
        raise ValueError(
            "batch、warmup、repeats 必须大于 0。"
        )

    sources = [str(p) for p in image_paths]

    print("\n========== FPS 性能测试 ==========")
    print(f"测试集图片数: {len(sources)}")
    print(f"输入尺寸: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"预热次数: {warmup}")
    print(f"正式测试次数: {repeats}")

    # --------------------------------------
    # 1. 预热，不计入 FPS
    # --------------------------------------

    print("正在预热模型...")

    for _ in range(warmup):
        results = model.predict(
            source=str(image_paths[0]),
            imgsz=imgsz,
            batch=1,
            device=device,
            half=False,
            stream=True,
            save=False,
            verbose=False,
        )

        # 必须消费生成器，才能真正执行预测
        for _ in results:
            pass

    _sync(device)

    # --------------------------------------
    # 2. 重复测试完整测试集
    # --------------------------------------

    fps_values = []

    for repeat in range(repeats):

        _sync(device)
        start = time.perf_counter()

        processed = 0

        results = model.predict(
            source=sources,
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=False,
            stream=True,
            save=False,
            verbose=False,
        )

        for _ in results:
            processed += 1

        # 确保 GPU 运算完成后再停止计时
        _sync(device)

        elapsed = time.perf_counter() - start

        if processed != len(sources):
            raise RuntimeError(
                f"预期处理 {len(sources)} 张图片，"
                f"实际处理 {processed} 张。"
            )

        fps = processed / elapsed
        fps_values.append(fps)

        print(
            f"第 {repeat + 1}/{repeats} 次: "
            f"{fps:.2f} FPS"
        )

    # --------------------------------------
    # 3. 汇总结果
    # --------------------------------------

    fps_mean = statistics.mean(fps_values)

    fps_std = (
        statistics.stdev(fps_values)
        if len(fps_values) > 1
        else 0.0
    )

    print("\n========== FPS 测试结果 ==========")
    print(f"FPS Mean: {fps_mean:.2f}")
    print(f"FPS Std:  {fps_std:.2f}")

    return fps_mean, fps_std
