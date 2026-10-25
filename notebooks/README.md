# Kaggle 训练 Notebook

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

1. 在 Kaggle 导入其中一个 `.ipynb`，启用 GPU 加速器，添加 `microning/ruod-r` 数据集。
2. 已设置数据根目录为 `/kaggle/input/datasets/microning/ruod-r/RUOD-R-YOLO-Split`，直接读取，不复制/解压、不重复校验数据。
3. 沿用你提供的 Python 3.13.15、PyTorch 2.11.0+cu128、Ultralytics 8.4.163 环境，不自动安装或升级。缺少库的新环境可取消 `%pip install ultralytics` 的注释。
4. 首次运行开启 Internet 下载预训练及库内部检查所需资源。已有权重可添加到 Input，并将 `PRETRAINED_WEIGHTS` 改为其完整路径。
5. 按顺序运行。Notebook 内嵌辅助函数及 P2 模型配置，无需上传项目源码。

默认 batch=16、workers=4、seed=42、patience=10、optimizer=auto。batch 和 workers 按当前要求调整，其余参数沿用原脚本。显存不足时减小 batch 至 8、4 或 2，并记录调整。类别配置来自 Input 的 `data.yaml`。你提供的 train/val/test 数量为 8608/956/4134，Notebook 不再扫描验证。

## GPU 使用

当前使用 GPU 0（单张 Tesla T4），第二张 GPU 不参与训练。环境检查会显示两张卡。保留单卡是为了让原有 `trainer.epoch`、优化器和 PyTorch 峰值显存统计继续正确工作；直接改为 `[0, 1]` 会启动独立训练子进程，现有父进程统计代码不能直接适用。训练后的评估与 FPS 也使用 GPU 0。

## 保存和恢复

所有生成文件保存在 `/kaggle/working/underwater_object_detection`：

- `runtime/data_kaggle.yaml`：适配 Kaggle 路径的数据配置；Input 中的原 YAML 不修改。
- `runs/<实验名>/`：训练结果、`weights/best.pt`、`weights/last.pt`、评估、预测图片和环境快照。
- `records/experiment_ledger.csv`：本次工作目录中的实验台账。

新训练自动避免覆盖同名实验。会话结束前请保存包含输出的 Notebook 版本，确认输出文件已保存，或下载结果。Working 不能当作跨会话的永久磁盘。最后一节可选打包当前实验及台账到 `/kaggle/working/<实验名>_results.zip`。

同一会话续训：设置 `RESUME_CHECKPOINT` 为 Working 中原实验的 `weights/last.pt`，再执行准备和训练步骤。

跨会话续训：把已保存的整个旧实验目录作为 Input 添加，填写 `RESTORE_RUN_FROM`，程序将它复制回原 Working 位置。必须包含 `weights/last.pt`、`weights/best.pt`、`results.csv`、`args.yaml`；仅恢复 last.pt 会丢历史曲线和已有最佳权重。若已恢复过，清空 `RESTORE_RUN_FROM`，直接指定 Working 中的 `RESUME_CHECKPOINT`。旧全局台账如需合并，请另行保存与整理。

保持原 Notebook、模型、数据和输出路径；续训的 batch、workers 使用当前配置（16、4），其余训练设置沿用断点。仅恢复尚未完成且保留优化器状态的训练，代码会拒绝已结束或不完整的断点，避免静默重开训练。续训耗时/显存统计只覆盖本次会话；`best_epoch` 留空以免误报全程最佳轮次，最佳权重仍使用 `best.pt`。

## 整理范围

- 将主函数拆成导入、环境检查、路径配置、参数、模型加载、训练、复杂度、验证、测试、预测图、FPS、台账等单元格。不重复进行标签格式及图片/标签对应检查。
- 内置 `experiment_ledger.py`、`inference_benchmark.py` 相关函数，环境检查采用 `check_env.py` 的检查项。
- 在 Kaggle Working 生成数据 YAML；P2 Notebook 内嵌原模型 YAML。不会在本地项目改写配置或脚本。
- 实验环境由你统一管理；实际 PyTorch、CUDA 和完整依赖另存为环境记录。
- Notebook 中预测/FPS 改用文件清单确保 batch=1 生效，避免原脚本 list 输入一次加载全部图片。FPS 测试的预热、重复次数及完整测试集范围不变，但旧列表输入测量结果与此处的 batch=1 结果不直接等价。
- 每个 Notebook 都包含中文说明，且执行计数和输出清空。测试集用于最终评估，不用于调参。

## 验证边界

已进行 Notebook 格式、全部代码单元格语法、原训练参数/P2 配置及 Kaggle 路径检查，并离线验证 YAML 生成和实验目录恢复。当前机器没有该 Kaggle GPU 会话，未执行真实 GPU 训练。

参考：[Kaggle Notebook 文档](https://www.kaggle.com/docs/notebooks)、[Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)、[训练器实现](https://github.com/ultralytics/ultralytics/blob/v8.4.163/ultralytics/engine/trainer.py)。
