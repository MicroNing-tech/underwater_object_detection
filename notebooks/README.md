# Google Colab 训练 Notebook

本目录将 `scripts/` 下 6 个训练脚本整理为对应的独立 Notebook。原 `scripts/` 内容保持不变。

| Notebook | 模型 | Epochs | 图像尺寸 |
| --- | --- | --- | --- |
| [yolo11n_e100_i640.ipynb](yolo11n_e100_i640.ipynb) | YOLO11n | 100 | 640 |
| [yolo11n_e100_i800.ipynb](yolo11n_e100_i800.ipynb) | YOLO11n | 100 | 800 |
| [yolo11n_e100_i960.ipynb](yolo11n_e100_i960.ipynb) | YOLO11n | 100 | 960 |
| [yolo11n_e50_i640.ipynb](yolo11n_e50_i640.ipynb) | YOLO11n | 50 | 640 |
| [yolo11s_e100_i800.ipynb](yolo11s_e100_i800.ipynb) | YOLO11s | 100 | 800 |
| [yolo11s_p2_e100_i800.ipynb](yolo11s_p2_e100_i800.ipynb) | YOLO11s-P2 | 100 | 800 |

## 使用方式

1. 在 [Google Colab](https://colab.research.google.com/) 选择“上传笔记本”，打开其中一个 `.ipynb`，并将运行时设为 GPU。
2. 将原数据集 `RUOD-R-YOLO-Split` 文件夹压缩为 ZIP 并上传 Drive。ZIP 内保留 `images/{train,val,test}` 和 `labels/{train,val,test}` 结构，也可直接使用 Drive 上的目录。
3. 修改 Notebook 第 2 节的 `DATA_SOURCE`（数据 ZIP 或目录）及 `DRIVE_PROJECT_ROOT`（结果保存目录）。默认数据路径为 `/content/drive/MyDrive/datasets/RUOD-R-YOLO-Split.zip`。
4. 按顺序运行。Notebook 内嵌辅助函数及 P2 模型配置，无需把项目源码上传到 Colab。

默认 batch=8、workers=0、seed=42、patience=10、optimizer=auto，与原脚本一致。资源不足可减小 batch，但应在比较实验时保持设置一致。保留 RUOD-R 原来的 10 个类别与编号。

## 保存和恢复

数据默认复制/解压到 `/content/datasets/ruod_r`；训练结果直接保存到 Drive 的 `underwater_object_detection/runs/`，台账写到 `records/experiment_ledger.csv`。新训练自动避免覆盖同名实验。

中断后重新执行准备步骤，将 `RESUME_CHECKPOINT` 设为同一实验的 `weights/last.pt`。保持原 Notebook、数据、Drive 输出路径和参数。只能恢复未完成且保留优化器状态的训练；完成后的权重应作为新训练的初始权重。续训耗时和显存统计只覆盖本次会话，已在环境记录中注明。续训后的 `best_epoch` 留空，避免将未恢复历史状态的 early-stopping 对象误当成全程最佳轮次；最佳权重仍由 `best.pt` 提供。

## 整理范围

- 将主函数拆成安装、环境检查、数据准备、参数、模型加载、训练、复杂度、验证、测试、预测图、FPS、台账等单元格。数据集已在本地检查，Notebook 直接使用，不重复进行标签格式及图片/标签对应检查。
- 内置 `experiment_ledger.py`、`inference_benchmark.py` 相关函数，环境检查采用 `check_env.py` 的检查项。
- 在 Colab 临时目录生成数据 YAML；P2 Notebook 内嵌原模型 YAML。不会在本地项目改写配置或脚本。
- 直接使用 `%pip install ultralytics` 安装，实验环境由你统一管理；实际 PyTorch、CUDA 和完整依赖另存为环境记录。
- Notebook 中预测/FPS 改用文件清单确保 batch=1 生效，避免原脚本 list 输入一次加载全部图片。FPS 测试的预热、重复次数及完整测试集范围不变，但旧列表输入测量结果与此处的 batch=1 结果不直接等价。
- 每个 Notebook 都包含中文说明，且执行计数和输出清空。测试集用于最终评估，不用于调参。

## 验证边界

已进行 Notebook 格式、全部代码单元格语法、原脚本参数/类别映射/P2 配置一致性检查，并对数据准备与辅助流程做离线验证。当前机器没有 PyTorch/Ultralytics，也没有 Colab GPU 会话，未进行真实 GPU 训练。

参考：[Colab FAQ](https://research.google.com/colaboratory/faq.html)、[Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)。
