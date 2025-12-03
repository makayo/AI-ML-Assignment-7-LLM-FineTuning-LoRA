# Sentiment Classification with LoRA
Author: MARK YOSINAO  
Artificial Intelligence — Parameter-Efficient Fine-Tuning, Large Language Models, and Output Evaluation

## Environment
- Python 3.9+
- Jupyter Notebook / JupyterLab (Anaconda environment)
- Hugging Face Transformers 4.57.3  
- PEFT 0.18.0 (LoRA support)  
- Torch 2.9.1 (CPU build, or GPU build w/ CUDA if capable)   
- Datasets 4.4.1  
- Scikit-learn 1.7.2  
- NumPy 2.3.5  
- Evaluate 0.4.6  

Note:** If your system supports CUDA, install the GPU-enabled build of PyTorch to accelerate training. 
---

## Project Overview
This project demonstrates the impact of **LoRA (Low-Rank Adaptation)** on fine-tuning a pretrained language model for sentiment classification.  
The task: classify IMDb movie reviews as **positive** or **negative**.  

By comparing a weak baseline (zero-shot T5-small) against a LoRA fine-tuned model, the workflow highlights how parameter-efficient fine-tuning can transform performance even on CPU.

---

## Dataset / Input
- **Dataset:** IMDb movie reviews (Hugging Face Datasets)  
- **Split:** Stratified into train, validation, and test sets  
- **Reduced for CPU debugging:**  
  - Train: 500 samples  
  - Validation: 200 samples  
  - Test: 100 samples  

Each review is formatted as:
- Input → `"review: <text>"`  
- Label → `"positive"` or `"negative"`  

---

## Workflow Summary
1. **Baseline Model (No Fine-Tuning)**  
   - Raw T5-small tested on validation set.  
   - Output: Predicted only “negative” for all reviews.  
   - Metrics: Accuracy 0.49, Precision 0.0, Recall 0.0, F1 0.0.  

2. **LoRA Configuration**  
   - Applied adapters to attention layers (`q`, `v`).  
   - r = 8, alpha = 32, dropout = 0.1.  
   - Trainable parameters: 294,912 (<1% of total).  

3. **Training**  
   - One epoch using Hugging Face Seq2SeqTrainer.  
   - Training loss: 0.1235, Validation loss: 0.0758.  

4. **Evaluation (Fine-Tuned Model)**  
   - Accuracy: 0.67  
   - Precision: 0.97  
   - Recall: 0.36  
   - F1: 0.53  
   - Class Breakdown:  
     - Negative → Precision 0.60, Recall 0.99, F1 0.75  
     - Positive → Precision 0.97, Recall 0.36, F1 0.53  

5. **Inference on Custom Reviews**  
   - “This movie was amazing!” → Positive  
   - “Terrible acting and a boring plot.” → Negative  
   - “The visuals were stunning, but the story was weak.” → Negative  

---

## Evaluation
| Model Type          | Accuracy | Precision | Recall | F1   | Notes                                                                 |
|---------------------|----------|-----------|--------|------|----------------------------------------------------------------------|
| Baseline (T5-small) | 0.49     | 0.0       | 0.0    | 0.0  | Predicted only “negative,” failed to recognize positives             |
| LoRA Fine-Tuned     | 0.67     | 0.97      | 0.36   | 0.53 | Strong negative recall, high positive precision, limited positive recall |

---

## Results
- **Baseline:** Failed to recognize positives, collapsed to predicting only negatives.  
- **LoRA Fine-Tuning:** Improved accuracy and enabled positive recognition.  
- **Strengths:** High precision for positives, strong recall for negatives.  
- **Limitations:** Positive recall remains low; struggles with mixed sentiment.  


---

## Why LoRA Saves Resources
LoRA saves computational resources by freezing the original model weights and introducing small trainable matrices into the attention layers.  
This reduces the number of trainable parameters to less than 1% of the full model, allowing fine-tuning on limited hardware (like CPU) while still achieving measurable improvements in performance.

This progression demonstrates how **LoRA fine-tuning enables efficient training and measurable performance gains**, even on CPU.

---

## Next Steps
- Train for more epochs to improve recall.  
- Balance dataset classes to reduce bias toward negatives.  
- Experiment with loss functions (e.g., focal loss).  
- Explore larger datasets or augmentation for positives.
