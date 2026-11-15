# TaSPNet 配套代码

本工程根据您提供的 TaSPNet 论文编写，包含 TANS、COSA、TNMB、PGTMB、冻结 SAM 先验、联合损失和完整训练/评估/预测入口。

**这是论文推导实现，不是作者原始实验代码。** 论文未明确的细节已列入 [IMPLEMENTATION.md](IMPLEMENTATION.md)。没有真实数据集、SAM 权重或训练好的道路模型，因此不能声称已复现文中精度。

## 先验证程序运行

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python demo.py
python train.py --config data/demo/config.json --train data/demo/train.json --val data/demo/val.json --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --manifest data/demo/test.json --output runs/demo/test.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --image data/demo/test_0.png --prior data/demo/test_0_prior.npy --output predictions/demo --device cpu
```

演示数据及随机先验只验证代码流程，不能作为 SAM 输出或论文实验结果。

## 使用真实数据

1. 按原始地理区域准备训练、验证、测试清单。每份清单为 JSON 列表，含 `image`、`mask`，可加 `group` 表示采集区域；文件路径相对于清单。标注只接受二值灰度 0/1 或 0/255，255 表示道路。
2. 安装 `requirements-sam.txt` 并从 SAM 官方仓库下载与模型类型匹配的权重。
3. 用 `prepare_priors.py` 对每个划分缓存先验。程序不读取标注，记录权重和图像的哈希值。
4. 用 `train.py` 训练，`evaluate.py` 在留出的测试集上评估，`predict.py` 导出原图尺寸概率和道路图。

```bash
python prepare_priors.py --manifest data/train_raw.json --output data/train.json --sam-checkpoint checkpoints/sam_vit_h.pth --sam-type vit_h
python prepare_priors.py --manifest data/val_raw.json --output data/val.json --sam-checkpoint checkpoints/sam_vit_h.pth --sam-type vit_h
python train.py --config config.json --train data/train.json --val data/val.json --output runs/real
```

测试集也需用同样的 SAM 模型和参数生成先验。默认输入为 512×512，AdamW 学习率 1e-4、权重衰减 1e-2、batch size 16，采用余弦调度。通道宽度、训练 100 轮、条带宽度、状态维度、损失权重等是暂定实现配置，并非论文明确报告值。

## 需要了解的实现边界

- 纯 PyTorch 后端实现真正的输入选择性状态递推，提供串行与并行等价性测试；没有用普通卷积冒充 Mamba。它是便于审查的参考实现，不能据此声称达到论文速度。
- 硬方向选择用 straight-through 代理梯度训练方向分支，排序本身不可微；这是补齐论文未交代训练方式的实现选择。
- SAM 按几何细长程度筛选候选区域，不保证每个候选区域都是道路。
- 公式（24）会产生约 0.5～0.73 的先验值，公式（27）的门控可能接近常数。当前保留原公式，建议作者确认后再决定是否修改。
- 默认全分辨率扫描在 512×512、batch 16 下可能消耗较多显存。当前没有验证 4090、多 GPU 训练或原始实验精度。
- 评价输出同时包括逐图平均和全局统计，均为百分比；不要把演示指标写入论文。

详细命令、断点续训和数据格式请见 [英文 README](README.md)。
