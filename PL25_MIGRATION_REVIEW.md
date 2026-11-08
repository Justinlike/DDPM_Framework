# VQGAN Lightning 2.5 检查记录

## 修改范围

- `models/vqgan_3d/vqgan.py`：切换 manual optimization；G→D 顺序不变，保留 reconstruction、commitment、LPIPS、adversarial、feature matching 全部原公式和权重。用 Lightning 包装的优化器、manual_backward 和 clip_gradients；toggle 隔离参数梯度。判别器输入原本已经 detach，保留。forward 中 optimizer_idx 是内部损失分支选择，并非 Lightning training_step 的旧接口，可以保留。
- 同文件：保留 G 更新后重新前向得到 D 输入的行为，避免复用旧重建而改变原策略；这也保留原 Codebook 每 batch 两次 EMA 更新。预热期间仍计算 D 指标但跳过 backward/step。单独保存旧优化器步计数（每完整 batch +2），保持延迟启动条件在原有计数尺度上；旧 checkpoint 回退到 global_step。Lightning 当前 global_step 仍是实际优化器 step 次数，因此预热期按 step 的保存频率/max_steps 与旧版有差异，不伪造 Lightning 状态。
- 同文件：恢复时关闭 Codebook 首批重新初始化；随机帧索引跟随输入设备；验证 LPIPS 取均值以确保批量大于 1 时日志为标量。
- `models/vqgan_3d/codebook.py`：仅将距离计算、one-hot 统计、EMA 和 perplexity 局部置于 FP32，公式及 EMA 系数不变。大体积统计和平方距离存在 FP16 溢出风险。
- `train_vqgan.py`：验证模态修正为 NCCT+CCTA，均 training=False；TorchIO Queue 外层 DataLoader 固定单进程。保留非配对 ConcatDataset。恢复路径传入 fit(ckpt_path=...)，显式路径优先，自动搜索覆盖 version_0、缺失 checkpoints 目录和嵌套路径，不重命名任何文件；自动选择最新版本目录中修改时间最新的 checkpoint，不保证它一定有最大训练 step。
- 同文件及 `conf/config-vqgan.yaml`：16 → 16-mixed；Trainer 不再配置自动梯度裁剪。现阶段仅支持 accumulate_grad_batches=1，其他值明确报错，避免静默错误。原学习率计算未动。

## 检查中无需修改的部分

Trainer accelerator/devices 已迁移。configure_optimizers 返回两优化器和空 scheduler 列表在手动模式可保留。没有学习率调度器需要恢复。未发现此路径的旧 epoch_end hook。val/recon_loss 确实存在。ModelCheckpoint 默认保存完整训练状态，保留名称、top-k 和两个周期回调。BaseDataset、数据集预处理、网络结构、损失权重、学习率、batch size、split 均未改。

## 仍需关注

1. 当前仅可访问本地副本；python3 -m pip show torch pytorch-lightning torchio 表明本地未安装这些依赖。服务器 2.5 版本来自用户说明，未远程探测，未安装或降级任何依赖。
2. 实际脚本硬编码 patch=(512,512,64)，YAML 为 (512,512,8)；实际使用 m11，YAML normalization 是另一个旧名称。为避免改变实际训练输入，本次未将这些配置接线或改值。配置 root_dir/train_split/val_split 也尚未接入硬编码的数据集创建。
3. train/test 目录不同，但无服务器数据，无法确认患者 ID 重叠或是否有独立 val；test 用于模型选择后不再是独立最终测试集。
4. Queue 长度 = 病例数 × samples_per_volume（默认16），queue_length=16 是缓存容量。train 522×16×2=16704；val 如每模态15例则15×16×2=480。training=False 不会让 UniformSampler 的 patch 位置固定；验证仍是随机 patch。
5. 学习率仍为 1×(1/8)×(1/4)×3e-4=9.375e-6，G/D 相同。公式形式是相对8卡、每卡batch4的线性缩放；原作者动机未独立确认。可能收敛较慢，需根据重建/量化/对抗指标评估，不在 API 迁移中擅改。
6. LPIPS 即使权重为0也会初始化并可能下载权重；未改变其行为。AMP数值、显存（512×512×64、n_hiddens=240）及多GPU DDP均需服务器实测。未加入额外 GradScaler/autocast 包装训练循环。
7. 旧1.x checkpoint 的 Lightning循环迁移、PyTorch2.7反序列化兼容性取决于真实文件，未做加载验证。不要对不可信checkpoint放宽反序列化限制。
8. 当前 max_epochs/max_steps 均为-1，正式配置可能无限训练；本次未改变训练时长。

## 已执行验证

- Python AST 解析训练入口、models/vqgan_3d 全部 Python 文件和数据集：通过。
- `python3 /tmp/check_vqgan_migration.py`：直接提取真实 training_step、更新和 checkpoint hooks，以测试替身检查预热只更新G、启动后G→D顺序、原损失求和、梯度裁剪调用、计数保存及旧checkpoint回退：通过。此检查不是实际反向传播测试。
- `git diff --check`：通过。
- 未执行：真实数据读取、完整模型forward、实际梯度/优化器状态变化、GPU AMP、1–3 batch训练/验证、真实checkpoint保存恢复。缺少本地依赖、服务器数据及GPU，不能标为通过。

## 服务器验证建议

先打印torch/lightning/torchio实际版本。使用独立临时输出目录、max_epochs=1、max_steps=3及limit_train_batches/limit_val_batches限制做小测试；自动恢复关闭，防止误读正式checkpoint。另用临时配置将discriminator_iter_start设为0测试D分支，不改正式配置。小batch测试通过后，在临时目录保存并恢复，检查两个optimizer_states、global_step/epoch、vqgan_legacy_optimizer_steps和Codebook缓冲区。训练入口当前未暴露limit_*参数，需在隔离测试脚本中直接构建Trainer，不能直接当作已有Hydra参数使用。

官方依据：https://lightning.ai/docs/pytorch/2.5.0/model/manual_optimization.html
