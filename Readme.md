# LiSalNet: Lightweight Saliency Detection Network for RGB-D Images

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)

</div>

## 📋 Overview

**LiSalNet** is an efficient lightweight framework for RGB-D salient object detection that achieves state-of-the-art performance while maintaining exceptional computational efficiency.

## 📊 Performance and Efficiency

**Evaluation on 6 Benchmark Datasets**: NJU2K, NLPR, STERE, DES, LFSD, and SIP. *More Results are coming soon*


### Additional Visual Insights (*More Visuals are coming soon*)

<div align="center">
  <img src="./performance_lines.png" alt=" Comparisons" width="95%">
</div>


## 🚀 Code & Resources

> **Note**: The complete source code will be publicly available after paper acceptance.



### Coming Soon (After Acceptance)
- ✅ **Pre-trained Models**: Will be released on the link [Download Models](link-to-models)
- ✅ **Visual Predictions**: Will be released on the link [Download Results on All Datasets](link-to-predictions)
  - NJU2K Dataset Predictions
  - NLPR Dataset Predictions
  - STERE Dataset Predictions
  - DES Dataset Predictions
  - LFSD Dataset Predictions
  - SIP Dataset Predictions
- 🔜 Complete training code
- 🔜 Inference scripts
- 🔜 Model architecture implementation
- 🔜 Preprocessing utilities
- 🔜 Evaluation metrics
- 🔜 Deployment guidelines

## 📦 Installation (Preview)
```bash
# Clone the repository (will be available after acceptance)
git clone https://github.com/username/LiSalNet.git
cd LiSalNet

# Create conda environment
conda create -n lisalnet python=3.8
conda activate lisalnet

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start (Preview)
```python
# Load pre-trained model
from models.lisalnet import LiSalNet

model = LiSalNet(pretrained=True)
model.eval()

# Inference on RGB-D image pair
saliency_map = model(rgb_image, depth_map)
```
## 🔧 Training (Coming Soon)
```bash
# Training script will be available after acceptance
python train.py --config configs/lisalnet_config.yaml
```

## 📊 Evaluation (Coming Soon)
```bash
# Evaluation script will be available after acceptance
python evaluate.py --dataset NJU2K --model_path checkpoints/lisalnet_best.pth
```

## 🎓 Citation

If you find LiSalNet useful in your research, please consider citing:
```bibtex
@article{lisalnet2025,
  title={LiSalNet: Lightweight Saliency Detection Network for RGB-D Images},
  author={Author Names},
  journal={Journal Name},
  year={}
}
```

## 📧 Contact

For questions and discussions, please contact:
- **Primary Author**: [habibkhan@ieee.org]
- **Lab**: Computational Imaging & Perception Lab, Gachon University

## 🙏 Acknowledgments

This work was supported by [Funding Information]. We thank the authors of the benchmark datasets for making their data publicly available.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>




