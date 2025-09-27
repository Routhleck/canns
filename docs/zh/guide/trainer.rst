Hebbian 记忆示例
==============

本章覆盖 ``examples/brain_inspired/`` 中的 Hopfield 网络示例，重点展示如何利用
:class:`HebbianTrainer <src.canns.trainer.HebbianTrainer>` 统一训练与预测流程，以及常见的数据预处理技巧。

``hopfield_train.py``
-------------------

- **路径**：`examples/brain_inspired/hopfield_train.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train.py>`_
- **场景**：从 ``skimage`` 自带图像生成训练模式，使用 Hebb 学习存储，再用噪声图像测试恢复能力。
- **关键步骤**：

  1. 调用 ``preprocess_image`` 将彩色/灰度图转换为 128×128 二值向量，并映射到 {-1,+1}。
  2. 初始化 :class:`AmariHopfieldNetwork <src.canns.models.brain_inspired.hopfield.AmariHopfieldNetwork>`（同步更新、sign 激活）。
  3. 创建 ``HebbianTrainer(model)`` 并执行 ``trainer.train(data_list)``。
  4. 使用 ``trainer.predict_batch`` 对随机翻转 30% 像素后的样本进行恢复。
  5. 通过 Matplotlib 排列“训练图/输入/输出”三列，可保存为 ``discrete_hopfield_train.png``。
- **扩展**：

  - 将 ``asyn=True`` 切换为异步更新，观察收敛差异。
  - 使用 ``normalize_by_patterns=False`` 比较是否保留模式间的幅值差异。

``hopfield_train_mnist.py``
------------------------

- **路径**：`examples/brain_inspired/hopfield_train_mnist.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train_mnist.py>`_
- **场景**：加载 MNIST（或其 fallback 数据集）的小批样本，演示如何选择代表性模式并执行 Hebb 学习。
- **关键步骤**：

  1. ``_load_mnist()`` 会按顺序尝试 HuggingFace datasets、TorchVision、Keras、scikit-learn，确保在离线环境也可运行。
  2. 选择 0–9 的少量样本作为训练模式，利用 ``_threshold_to_pm1`` 转换为 {-1,+1}。
  3. 初始化 Hopfield 网络并执行 ``trainer.train(patterns)``。
  4. 对干净的测试样本调用 ``trainer.predict``，验证记忆检索。
  5. 通过 ``plt.subplots`` 将 Train/Input/Output 绘制到单页图像。
- **扩展**：

  - 准备噪声版本的测试图像，评估重建鲁棒性。
  - 使用 ``trainer.configure_progress(show_iteration_progress=True)`` 查看预测时的能量变化。

自定义方向
----------

- 想要接入自定义模型，仅需遵循 :class:`BrainInspiredModel <src.canns.models.brain_inspired.BrainInspiredModel>`
  的接口（暴露 ``W``、``s``、``update()``、``energy`` 即可）。
- 若要组合调参，可在脚本最上方调整 ``AmariHopfieldNetwork`` 的 ``activation``、``temperature``
  等参数，并观察对能量收敛速度的影响。
