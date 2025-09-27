示例索引
========

本章给出一个全局索引，便于根据目标快速定位脚本：

- **入门体验**：想要最快看到 CANN 的运行效果，可先浏览 :doc:`models` 中的
  `cann1d_oscillatory_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_oscillatory_tracking.py>`_ 与 `cann2d_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann2d_tracking.py>`_，二者分别展示 1D/2D bump 的生成、追踪与动图导出。
- **Hebb 学习与记忆**：若关注 Hopfield 或脑启发模型，请转到 :doc:`trainer`，使用
  `hopfield_train.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train.py>`_ 与 `hopfield_train_mnist.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train_mnist.py>`_ 学习如何调用统一的 :class:`HebbianTrainer <src.canns.trainer.HebbianTrainer>`。
- **导航与任务扩展**：对于空间路径、外部轨迹导入、层次路径积分等主题，可参考 :doc:`tasks`。
- **实验数据分析**：需要加载 Hugging Face 数据集并执行拟合/拓扑分析时，详见 :doc:`analyzer`。
- **自动化流水线**：完整的 theta sweep 数据流处理（含动画导出）集中在 :doc:`pipeline`。

运行准备
--------

1. **创建虚拟环境**：
   
   .. code-block:: bash

      make install

2. **运行脚本**：建议通过 ``uv`` 调用，保证依赖一致（以 `cann1d_oscillatory_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_oscillatory_tracking.py>`_ 为例）：
   
   .. code-block:: bash

      uv run python examples/cann/cann1d_oscillatory_tracking.py

3. **查看输出**：大部分示例会在当前目录生成图像/动画/npz 文件，名称可在脚本结尾处找到。

Notebook 入口
----------------

除了 Python 脚本，``docs/zh/notebooks/01_quick_start.ipynb`` 与 ``00_design_philosophy.ipynb``
提供了互动式教程。若需要在线体验，可使用 README 中的 Binder / Colab 链接预装环境。

继续阅读
--------

接下来章（:doc:`models` 到 :doc:`pipeline`）以“脚本 → 思路 → 关键 API → 延伸”四个部分展开。
你可以按需挑选感兴趣的示例，或顺序学习以了解 CANNs 项目的常见工作流。
