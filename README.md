<h1 align="center"> HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos </h1>



<div align="center">

**CoRL 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2602.02473-b31b1b.svg)](https://arxiv.org/abs/2602.02473)
[![Project Page](https://img.shields.io/badge/Project-Page-4c9be8.svg)](https://wyhuai.github.io/human-x/)

[English](README.md) | [简体中文](README.zh-CN.md)
</div>

---


# What You Can Reproduce

The released code and example data support reproducing the following
humanoid-object interaction skills in simulation:

- 🏀 Basketball catch-and-throw
- 🏀 Basketball pump-fake shot
- ⚽️ Football kicking
- 📦 Bin pick-and-place
- 💡 New interaction skills learned from your own videos

# Overview

HumanX is a full-stack framework for turning human videos into agile and
generalizable humanoid interaction skills.

The pipeline consists of two complementary components:

- **XGen**: a motion and HOI editor that turns compatible robot motions into
  physically plausible humanoid-object interaction motions.
- **XMimic**: a unified imitation learning framework for training and
  evaluating interaction skills in simulation.

The interface between the two components is motion data. XGen edits and
exports robot and HOI `.npz` files, while XMimic converts compatible XGen
outputs into the HumanX `.pkl` format used for training and evaluation.

For new skills, the recommended video-to-policy pipeline is:

1. **[GVHMR](https://github.com/zju3dv/GVHMR)** recovers world-grounded human
   motion from monocular video.
2. **[UMR](https://github.com/hanyang9/UMR)** retargets the recovered motion to
   the G1 humanoid.
3. **[XGen](xgen/README.md)** edits and synthesizes physically plausible
   humanoid-object interaction motions, then exports HOI `.npz` files.
4. **[XMimic](xmimic/README.md)** converts the XGen HOI `.npz` files into HumanX `.pkl` files and
   trains the corresponding interaction policy.

## Repository Structure

```text
HumanX/
  xgen/                         XGen HOI editing, conversion, and data docs
    xgen_editor/                XGen editor and conversion tools
    data/                       Data format and quality documentation
  xmimic/                       Simulation-side imitation learning
    humanoidverse/              Training, evaluation, environments, and configs
    tools/                      Dataset conversion and visualization tools
    data/                       Local motion and asset data
```

## Documentation

Use the README for the component you want to run:

- [XGen README](xgen/README.md): XGen editor, motion formats,
  HOI synthesis, visualization, and conversion tools.
- [XMimic README](xmimic/README.md): installation, training, evaluation,
  XGen-to-XMimic conversion, and motion playback.

## End-to-End Workflow

1. Use [XGen](xgen/README.md) to inspect robot motion, synthesize
   HOI data, and export motion `.npz` files.
2. Follow the [XMimic data pipeline](xmimic/README.md#xgen-to-xmimic-data-pipeline)
   to convert XGen HOI data into HumanX `.pkl` motion data.
3. Use the [XMimic training and evaluation instructions](xmimic/README.md#training)
   to train, evaluate, and visualize the resulting policy.

For the standard interface, keep source XGen data and generated XMimic motion
files under the corresponding `data/` directories.


# Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026humanx,
  title={HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos},
  author={Wang, Yinhuai and Zhao, Qihan and Lau, Yuen Fui and Yu, Runyi and Tsui, Hok Wai and Chen, Qifeng and Wang, Jingbo and Pang, Jiangmiao and Tan, Ping},
  journal={arXiv preprint arXiv:2602.02473},
  year={2026}
}
```

# Acknowledgements

This repository builds upon prior open-source efforts, including:

- [ASAP](https://github.com/LeCAR-Lab/ASAP)
- [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse)
- [BeyondMimic](https://beyondmimic.github.io/)
- [SkillMimic](https://github.com/wyhuai/SkillMimic)

We thank these communities for their excellent work.

# License

This project is licensed under the Apache-2.0 WITH Commons-Clause
