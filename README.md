# ExPO: Explainable Phonetic Trait-Oriented Network for Speaker Verification

Welcome to the official repository of **ExPO**: an Explainable Phonetic Trait-Oriented Network for speaker verification. This model introduces a novel approach to enhance explainability in speaker verification by incorporating phonetic traits, bridging the gap between manual forensic voice comparison and neural speaker verification systems.

---

## 📄 Abstract

In speaker verification, achieving explainability akin to forensic voice comparison has remained a challenge. ExPO leverages phonetic traits to generate utterance-level speaker embeddings and enables fine-grained analysis and visualization of phonetic traits. This explainable framework enhances trust and transparency while maintaining robust speaker verification performance.

For detailed insights, refer to the [paper](https://arxiv.org/abs/2501.05729).

---

## 🌟 Features

- **Explainable Verification**: Fine-grained phonetic trait analysis provides a transparent decision-making process.
- **State-of-the-Art Architecture**: Built on the ECAPA-TDNN backbone with integrated phonetic trait layers.
- **Custom Loss Functions**:
  - **Trait Verification Loss**: Ensures consistency of phonetic traits within the same speaker.
  - **Trait Center Loss**: Aligns phonetic traits across utterances for better generalization.
  - **Additive Angular Margin Loss (AAM)**: Enhances discriminability of speaker embeddings.
- **Compatibility**: Trained and tested on benchmark datasets including VoxCeleb and LibriSpeech.

---

## 📊 Performance

| **Model**                  | **EER (%)** | **minDCF** | **Explainability (EVD)** |
|----------------------------|-------------|------------|--------------------------|
| ECAPA-TDNN (Baseline)      | 1.276       | 0.157      | Limited                 |
| ExPO (Full)                | 1.552       | 0.184      | High                    |

---

## 🔧 Installation
1. **Prepare data**:
   The GitHub repository [charsiu](https://github.com/lingjzhu/charsiu) was used to generate phoneme files. We utilized the Textless Alignment method to generate the phoneme files.
   
   The pipeline for preparing speech samples in this repository is the same as that used in [WeSpeaker](https://github.com/wenet-e2e/wespeaker).
   
   Dataset for training usage:
   VoxCeleb1、2 training set;
   MUSAN dataset;
   RIR dataset.
   
3. **Dependencies**:
   ```bash
   git clone https://github.com/mmmmayi/ExPO.git
   cd ExPO
   pip install -r requirements.txt
   
4. **Training**:
   ```bash
   cd examples/voxceleb/v2
   ./run.sh
   
## 📚 Citation

If you find this project useful in your research, please consider citing our paper:

```bibtex
 @misc{ma2025expoexplainablephonetictraitoriented,
      title={ExPO: Explainable Phonetic Trait-Oriented Network for Speaker Verification}, 
      author={Yi Ma and Shuai Wang and Tianchi Liu and Haizhou Li},
      year={2025},
      eprint={2501.05729},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.05729}, 
 }
```

## 🙏 Acknowledgements

This project builds upon and is inspired by the work of several open-source repositories. We extend our gratitude to the authors and contributors of the following projects:

[charsiu](https://github.com/lingjzhu/charsiu)

[WeSpeaker](https://github.com/wenet-e2e/wespeaker)

[ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

[voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer)

Thanks for these authors to open source their code!
