# X-Ray Vision-Language Model (VLM)

## Objective
Develop a Vision-Language model that diagnoses pneumonia from chest X-rays and improves accuracy using textual clinical context.

## Dataset
Chest X-Ray Pneumonia Dataset (Kaggle)  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Model Architecture
- Image Encoder: ResNet18 (pretrained)
- Text Encoder: TF-IDF (synthetic clinical context)
- Fusion: Feature concatenation
- Classifier: Fully connected layer

## Experiments
| Model | Accuracy |
|------|---------|
| Image-only CNN | ~82–85% |
| Image + Text (VLM) | ~86–88% |

Textual context improved diagnostic confidence and reduced false positives.

## How to Run
```bash
pip install -r requirements.txt
python src/train_baseline.py
python src/train_fusion.py
python src/inference.py
