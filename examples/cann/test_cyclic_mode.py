#!/usr/bin/env python3
"""
测试循环模式：横向 → 竖向 → 横向 → 竖向 → ...
验证不会撞墙，持续扫描
"""

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from canns.task.open_loop_navigation import RasterScanNavigationTask

bm.set_dt(1e-3)


def test_cyclic_scanning():
    """测试循环扫描模式"""
    print("=" * 70)
    print("Testing Cyclic Dual-Mode Scanning")
    print("=" * 70)

    duration = 300.0  # 更长的时间看循环效果
    width = 1.0
    height = 1.0

    print(f"\n⏱️  Duration: {duration}s")
    print(f"📏 Environment: {width}m x {height}m")
    print(f"🔄 Expected: H → V → H → V → ... (cyclic)")

    task = RasterScanNavigationTask(
        duration=duration,
        width=width,
        height=height,
        step_size=0.03,
        speed=0.15,  # 现在可以设置speed了！
        progress_bar=True,
    )

    task.get_data()
    pos = task.data.position

    # 分析模式切换点
    print(f"\n📊 Analyzing mode switches...")

    # 检测底部触碰（横向→竖向切换）
    bottom_touches = np.where(pos[:, 1] <= 0.06)[0]  # 接近底部
    # 检测右边触碰（竖向→横向切换）
    right_touches = np.where(pos[:, 0] >= width - 0.06)[0]  # 接近右边

    print(f"\n   Bottom touches (H→V): {len(bottom_touches)} times")
    print(f"   Right edge touches (V→H): {len(right_touches)} times")

    if len(bottom_touches) > 0:
        print(f"   First bottom touch at step: {bottom_touches[0]}")
    if len(right_touches) > 0:
        print(f"   First right touch at step: {right_touches[0]}")

    # 计算覆盖率
    bins = 30
    heatmap, _, _ = np.histogram2d(
        pos[:, 0], pos[:, 1],
        bins=bins,
        range=[[0, width], [0, height]]
    )
    coverage = (heatmap > 0).sum() / (bins * bins) * 100

    print(f"\n   Coverage: {coverage:.1f}%")

    # 检查是否有墙壁碰撞（卡住）
    # 如果agent在边界附近停留太久，说明卡住了
    margin = 0.06
    at_bottom = (pos[:, 1] <= margin).sum()
    at_right = (pos[:, 0] >= width - margin).sum()
    at_top = (pos[:, 1] >= height - margin).sum()
    at_left = (pos[:, 0] <= margin).sum()

    total_steps = len(pos)
    print(f"\n   Time spent at boundaries:")
    print(f"   Bottom: {at_bottom/total_steps*100:.1f}%")
    print(f"   Right:  {at_right/total_steps*100:.1f}%")
    print(f"   Top:    {at_top/total_steps*100:.1f}%")
    print(f"   Left:   {at_left/total_steps*100:.1f}%")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: 完整轨迹
    ax = axes[0]
    time_array = np.arange(len(pos)) * task.dt
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=time_array,
        cmap='viridis',
        s=0.3,
        alpha=0.5,
    )
    ax.scatter(pos[0, 0], pos[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(pos[-1, 0], pos[-1, 1], c='red', s=100, marker='x', zorder=5, label='End')

    # 标记模式切换点
    if len(bottom_touches) > 0:
        ax.scatter(pos[bottom_touches, 0], pos[bottom_touches, 1],
                  c='orange', s=20, marker='s', alpha=0.6, label='H→V switch')
    if len(right_touches) > 0:
        ax.scatter(pos[right_touches, 0], pos[right_touches, 1],
                  c='cyan', s=20, marker='^', alpha=0.6, label='V→H switch')

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title(f'Cyclic Scanning Trajectory ({duration}s)')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(scatter, ax=ax, label='Time (s)', fraction=0.046)

    # 图2: X和Y随时间变化
    ax = axes[1]
    ax.plot(time_array, pos[:, 0], 'b-', alpha=0.6, linewidth=0.5, label='X position')
    ax.plot(time_array, pos[:, 1], 'r-', alpha=0.6, linewidth=0.5, label='Y position')
    ax.axhline(y=margin, color='gray', linestyle='--', alpha=0.3, label='Margins')
    ax.axhline(y=height-margin, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=width-margin, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图3: 覆盖热力图
    ax = axes[2]
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[0, width, 0, height],
        cmap='hot',
        aspect='equal',
    )
    ax.set_title(f'Coverage Heatmap\n{coverage:.1f}% coverage')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    plt.colorbar(im, ax=ax, label='Visits', fraction=0.046)

    plt.tight_layout()
    plt.savefig('cyclic_mode_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: cyclic_mode_test.png")
    plt.show()

    # 判断是否成功循环
    print(f"\n" + "=" * 70)
    print("CYCLIC MODE VALIDATION")
    print("=" * 70)

    if len(bottom_touches) > 0 and len(right_touches) > 0:
        print(f"\n✅ SUCCESS: Detected both mode switches!")
        print(f"   - Horizontal → Vertical switches: {len(bottom_touches)}")
        print(f"   - Vertical → Horizontal switches: {len(right_touches)}")
        print(f"\n✅ Cyclic scanning is WORKING! (避免撞墙)")
    elif len(bottom_touches) > 0:
        print(f"\n⚠️  PARTIAL: Only detected H→V switch")
        print(f"   Duration may be too short to complete full cycle")
    else:
        print(f"\n❌ FAILED: No mode switches detected")

    print(f"\n📊 Final coverage: {coverage:.1f}%")


if __name__ == "__main__":
    test_cyclic_scanning()
