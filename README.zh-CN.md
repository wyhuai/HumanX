<h1 align="center">HumanX：从人类视频学习敏捷且可泛化的人形机器人交互技能</h1>



<div align="center">

**CoRL 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2602.02473-b31b1b.svg)](https://arxiv.org/abs/2602.02473)
[![Project Page](https://img.shields.io/badge/Project-Page-4c9be8.svg)](https://wyhuai.github.io/human-x/)

[English](README.md) | [简体中文](README.zh-CN.md)
</div>

---

# 目录

- [可以复现什么](#可以复现什么)
- [项目概览](#项目概览)
- [仓库结构](#仓库结构)
- [文档](#文档)
- [完整流程](#完整流程)
- [引用](#引用)

# 可以复现什么

当前发布的代码和示例数据支持在仿真中复现以下人形机器人与物体交互技能：

- 🏀 篮球接球与投掷
- 🏀 篮球虚晃后仰跳投
- ⚽ 足球动态踢球
- 📦 搬运箱子
- 💡 使用自己的视频生成并学习新的交互技能

# 项目概览

HumanX 是一个将人类视频转换为敏捷且可泛化的人形机器人交互技能的完整框架。

整个流程由两个互补的组件组成：

- **XGen**：动作和人机物交互（HOI）编辑器，将兼容的机器人动作转换为物理上合理的人形机器人与物体交互动作。
- **XMimic**：统一的模仿学习框架，在仿真中训练和评估交互技能。

两个组件之间通过 motion data 连接。XGen 编辑并导出机器人和 HOI `.npz`
文件，XMimic 将兼容的 XGen 输出转换为用于训练和评估的 HumanX `.pkl`
格式。

对于新的交互技能，推荐使用以下从视频到策略的流程：

1. **[GVHMR](https://github.com/zju3dv/GVHMR)**：从单目视频恢复世界坐标系下的人体运动。
2. **[UMR](https://github.com/hanyang9/UMR)**：将恢复的人体运动 retarget 到 G1 人形机器人。
3. **[XGen](xgen/README.md)**：编辑并合成物理合理的人形机器人与物体交互动作，导出 HOI `.npz` 文件。
4. **[XMimic](xmimic/README.md)**：将 XGen HOI `.npz` 文件转换为 HumanX `.pkl` 文件，并训练对应的交互策略。

## 仓库结构

```text
HumanX/
  xgen/                         XGen HOI 编辑、转换和数据文档
    xgen_editor/                XGen 编辑器和转换工具
    data/                       XGen 示例 motion 数据
  xmimic/                       仿真侧模仿学习代码
    humanoidverse/              训练、评估、环境和配置
    tools/                      数据集转换和可视化工具
    data/                       本地 motion 和 asset 数据
```

## 文档

请根据要运行的组件阅读对应 README：

- [XGen README](xgen/README.md)：XGen 编辑器、动作格式、HOI 合成、可视化和转换工具。
- [XMimic README](xmimic/README.md)（英文）：安装、训练、评估、XGen 到 XMimic 的数据转换和 motion playback。

## 完整流程

1. 使用 [XGen](xgen/README.md) 检查机器人动作、合成 HOI 数据，并导出 `.npz` motion 文件。
2. 按照 [XMimic 数据管线](xmimic/README.md#xgen-to-xmimic-data-pipeline)，将 XGen HOI 数据转换为 HumanX `.pkl` motion 数据。
3. 按照 [XMimic 训练和评估说明](xmimic/README.md#training)训练、评估并可视化策略。

对于标准数据接口，请将 XGen 源数据和 XMimic 生成的 motion 文件放在各自
组件的 `data/` 目录下。

# 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{wang2026humanx,
  title={HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos},
  author={Wang, Yinhuai and Zhao, Qihan and Lau, Yuen Fui and Yu, Runyi and Tsui, Hok Wai and Chen, Qifeng and Wang, Jingbo and Pang, Jiangmiao and Tan, Ping},
  journal={arXiv preprint arXiv:2602.02473},
  year={2026}
}
```

# 致谢

本项目基于以下开源项目：

- [ASAP](https://github.com/LeCAR-Lab/ASAP)
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse)
- [BeyondMimic](https://beyondmimic.github.io/)
- [SkillMimic](https://github.com/wyhuai/SkillMimic)

感谢这些项目和社区的贡献。

# 许可证

本项目采用 Apache-2.0 WITH Commons-Clause 许可证。
