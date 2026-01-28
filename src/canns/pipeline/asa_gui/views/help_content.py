"""Help content for ASA GUI."""

from __future__ import annotations


def preprocess_help_markdown(lang: str = "en") -> str:
    if str(lang).lower().startswith("zh"):
        return (
            "# Preprocess 说明\n"
            "\n"
            "- **输入格式**：仅支持 ASA `.npz`，需要包含 `spike`, `x`, `y`, `t`。\n"
            "- **Embed spike trains**：将脉冲事件嵌入为密集矩阵，供 TDA / FR / FRM 使用。\n"
            "\n"
            "## Embed 参数\n"
            "- `res, dt`：时间分箱/步长，需与 `t` 的单位一致。\n"
            "- `sigma`：平滑尺度（越大越平滑，但会模糊快变化）。\n"
            "- `smooth`：是否启用平滑。\n"
            "- `speed_filter` + `min_speed`：过滤低速时间点（常见于 grid cell 数据）。\n"
            "  启用后解码会生成 `times_box`，后续 CohoMap/PathCompare/CohoSpace\n"
            "  需要用 `times_box` 对齐坐标与 `x/y/t`。\n"
        )
    return (
        "# Preprocess Guide\n"
        "\n"
        "- **Input format**: ASA `.npz` with `spike`, `x`, `y`, `t`.\n"
        "- **Embed spike trains**: build a dense matrix for TDA / FR / FRM.\n"
        "\n"
        "## Embed parameters\n"
        "- `res, dt`: time bin / step size (same unit as `t`).\n"
        "- `sigma`: smoothing scale (larger → smoother).\n"
        "- `smooth`: enable smoothing.\n"
        "- `speed_filter` + `min_speed`: remove low-speed samples (common for grid data).\n"
        "  When enabled, decoding yields `times_box`; downstream plots must align with it.\n"
    )


def analysis_help_markdown(mode: str | None, lang: str = "en") -> str:
    if str(lang).lower().startswith("zh"):
        base = (
            "# ASA 分析流程\n"
            "\n"
            "TDA → 解码 → CohoMap / PathCompare / CohoSpace / FR / FRM\n"
            "\n"
            "- 若 preprocess 启用了 `speed_filter`，解码会输出 `times_box`。\n"
            "  绘图时需用 `times_box` 对齐，否则可能出现长度不匹配。\n"
        )
        mode_text = {
            "tda": (
                "## TDA 参数\n"
                "- `num_times`：时间下采样步长（越大越快，但可能丢细节）。\n"
                "- `active_times`：选取最活跃时间点数（过小不稳，过大更慢/更噪）。\n"
                "- `dim`：PCA 维度（常见起点 6–12）。\n"
                "- `k`, `n_points`, `nbs`：采样/邻域构建相关，影响速度与稳定性。\n"
                "- `metric`：推荐 `cosine`。\n"
                "- `maxdim`：建议先用 1 确认环结构，再尝试 2 看 torus。\n"
                "- `coeff`：有限域系数（默认 47）。\n"
                "- `Shuffle`：用于显著性检验，代价高，建议少量。\n"
            ),
            "cohomap": (
                "## CohoMap\n"
                "- 将解码得到的相位轨迹投影回真实 `x/y` 轨迹。\n"
                "- 若启用 `speed_filter`，请使用 `coordsbox/times_box` 对齐。\n"
            ),
            "pathcompare": (
                "## PathCompare\n"
                "- 对比真实路径与解码路径的对齐效果。\n"
                "- `dim_mode`：1D/2D；`dim/dim1/dim2` 对应解码维度。\n"
                "- `use_box`：使用 `coordsbox/times_box` 对齐（speed_filter 时建议开启）。\n"
                "- `tmin/tmax` 或 `imin/imax`：截取区间。\n"
                "- `stride`：抽样步长；`tail`：尾迹长度；`fps`：动画帧率。\n"
            ),
            "cohospace": (
                "## CohoSpace / CohoScore\n"
                "- 查看神经元在相位空间中的偏好分布。\n"
                "- `top_percent`：选取活跃点百分比。\n"
                "- `view`：single neuron / population；`top_k` 仅对 population 生效。\n"
                "- `use_best`：使用 CohoScore 最小（更集中）的 neuron。\n"
                "- `unfold=skew`：在 skewed torus 平铺展示（2D 才可用）。\n"
            ),
            "fr": (
                "## FR Heatmap\n"
                "- 群体放电热图。\n"
                "- `neuron_range` / `time_range`：裁剪范围。\n"
                "- `normalize`：归一化方式；`none` 表示不归一化。\n"
                "- `mode=fr` 需要 preprocess；`spike` 可直接用事件。\n"
            ),
            "frm": (
                "## FRM\n"
                "- 单神经元空间放电图。\n"
                "- `neuron_id`：要看的神经元。\n"
                "- `bin_size`：空间分箱大小；`min_occupancy`：最小占据数。\n"
                "- `smoothing/smooth_sigma`：平滑选项。\n"
            ),
            "gridscore": (
                "## GridScore\n"
                "- 基于自相关图计算 gridness。\n"
                "- `bins`：空间分箱数；`min_occupancy`：最小占据。\n"
                "- `smoothing/sigma`：平滑选项（需 scipy）。\n"
                "- `overlap`：自相关重叠比例。\n"
                "- `mode=fr` 需要 preprocess；`spike` 可直接用事件。\n"
            ),
        }
        return f"{base}\n{mode_text.get(str(mode or ''), '')}".strip()

    base = (
        "# ASA Analysis Flow\n"
        "\n"
        "TDA → decode → CohoMap / PathCompare / CohoSpace / FR / FRM\n"
        "\n"
        "- If `speed_filter` is enabled in preprocess, decoding yields `times_box`.\n"
        "  Downstream plots must align with `times_box` to avoid length mismatches.\n"
    )

    mode_text = {
        "tda": (
            "## TDA parameters\n"
            "- `num_times`: time downsampling step (larger is faster but less detail).\n"
            "- `active_times`: number of most active points (too small = unstable).\n"
            "- `dim`: PCA dimension (typical 6–12).\n"
            "- `k`, `n_points`, `nbs`: sampling / neighborhood parameters.\n"
            "- `metric`: distance metric (recommend `cosine`).\n"
            "- `maxdim`: start with 1; try 2 for torus.\n"
            "- `coeff`: finite field coefficient (default 47).\n"
            "- `Shuffle`: significance test; expensive, keep small.\n"
        ),
        "cohomap": (
            "## CohoMap\n"
            "- Project decoded phase back to real `x/y` trajectory.\n"
            "- If `speed_filter` is on, use `coordsbox/times_box` for alignment.\n"
        ),
        "pathcompare": (
            "## PathCompare\n"
            "- Compare decoded vs real trajectory.\n"
            "- `dim_mode`: 1D/2D; `dim/dim1/dim2` select decoded dimensions.\n"
            "- `use_box`: align with `coordsbox/times_box` (recommended with speed_filter).\n"
            "- `tmin/tmax` or `imin/imax`: crop range.\n"
            "- `stride`: sampling step; `tail`: trail length; `fps`: animation rate.\n"
        ),
        "cohospace": (
            "## CohoSpace / CohoScore\n"
            "- Visualize neuron preference in phase space.\n"
            "- `top_percent`: active percentile threshold.\n"
            "- `view`: single neuron / population; `top_k` for population only.\n"
            "- `use_best`: pick neuron with smallest CohoScore (most selective).\n"
            "- `unfold=skew`: skewed torus tiling (2D only).\n"
        ),
        "fr": (
            "## FR Heatmap\n"
            "- Population firing-rate heatmap.\n"
            "- `neuron_range` / `time_range`: crop.\n"
            "- `normalize`: normalization method; `none` = no normalization.\n"
            "- `mode=fr` requires preprocess; `spike` uses events directly.\n"
        ),
        "frm": (
            "## FRM\n"
            "- Single-neuron spatial firing map.\n"
            "- `neuron_id`: target neuron.\n"
            "- `bin_size`: spatial bins; `min_occupancy`: minimum occupancy.\n"
            "- `smoothing/smooth_sigma`: smoothing options.\n"
        ),
        "gridscore": (
            "## GridScore\n"
            "- Compute gridness from autocorrelation.\n"
            "- `bins`: spatial bin count; `min_occupancy`: minimum occupancy.\n"
            "- `smoothing/sigma`: smoothing (requires scipy).\n"
            "- `overlap`: autocorr overlap ratio.\n"
            "- `mode=fr` requires preprocess; `spike` uses events directly.\n"
        ),
    }

    return f"{base}\n{mode_text.get(str(mode or ''), '')}".strip()
