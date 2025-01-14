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
   
2. **Dependencies**:
   ```bash
   git clone https://github.com/mmmmayi/ExPO.git
   cd ExPO
   pip install -r requirements.txt
3.
