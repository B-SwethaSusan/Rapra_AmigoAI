## Model Card for Sentiment Analysis

### Model Details
**Model Type:** Transformer-based sequence classification
**Architecture:** DistilBERT (a smaller, faster version of BERT with 6 layers, 768 hidden size, 12 attention heads) + classification head (linear layer with softmax over 2 classes)
**Fine-tuning Task:** Sentiment classification (binary: Positive vs Negative)
**Languages:** English (DistilBERT was pre-trained on English corpora like Wikipedia + Toronto BookCorpus)

### Training Data
**Dataset:** Custom sentiment analysis dataset (movie/product review–style text)
**Training Split:** 70% of total data
**Validation Split:** 15% of total data (X_dev in your code)
**Test Split:** 15% of total data (held-out set)
**Class Distribution:** Balanced binary classes (equal proportion of positive and negative samples)

### Evaluation
- **Evaluation Metrics**:
  - Accuracy: 97.9%
  - F1 Score: 0.9796
  - Validation Loss: 0.0546

### Training Configuration and Hyperparameters
- **Framework**: PyTorch with HuggingFace Transformers
- **Hardware**: 1x T4 GPU (Google Colab)
- **Total epoches**: 15
- **Training Time**: ~5 epochs (early stopping)
- **Batch Size**: 6
- **Learning Rate**: 2e-5
- **Weight Decay**: 0.01
- **Max Sequence Length**: 128 tokens
- **Optimizer**: AdamW

### Model Performance
**Validation Result**
| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.6915        | 0.6782          | 50.00%   | 0.6667   |
| 2     | 0.5929        | 0.1890          | 95.83%   | 0.9600   |
| 3     | 0.0959        | 0.0546          | 97.92%   | 0.9796   |
| 4     | 0.0092        | 0.1144          | 97.92%   | 0.9796   |
| 5     | 0.0045        | 0.1240          | 97.92%   | 0.9796   |

### Baseline versus improved approach with numbers on dev

### Model Comparison

| Feature        | Baseline (TF-IDF) | DistilBERT | Improvement |
|----------------|-------------------|------------|-------------|
| Accuracy       | 75.0%             | 97.92%     | +30.5%      |
| F1 Score       | 0.76              | 0.98       | +28.9%      |
| Training Time  | 2min              | 15min      | +650%       |
| Inference Speed| 1000 docs/s       | 200 docs/s | -80%        |
| Model Size     | 5MB               | 260MB      | +5100%      |

### Error Analysis
1. **Error Type 1**: Mixed Sentiment Analysis
   - **Example**: "Great battery life but overheats quickly"
   - **Hypothesis**: Model struggles with sentences containing both positive and negative sentiments
   - **Potential Fix**: Implement aspect-based sentiment analysis

2. **Error Type 2**: Sarcasm and Irony
   - **Example**: "Oh great, another update that breaks everything"
   - **Hypothesis**: Model fails to detect sarcastic tone
   - **Potential Fix**: Include more sarcastic examples in training data

3. **Error Type 3**: Domain-Specific Language
   - **Example**: "The latency is under 100ms which is acceptable for our use case"
   - **Hypothesis**: Technical terms may be misinterpreted
   - **Potential Fix**: Domain-specific fine-tuning

4. **Error Type 4**: Ambiguous or Weak Sentiment
   - **Example**: "Packaging feels ok"
   - **Hypothesis**: Model struggles with mild and ambiguous sentiment
   - **Potential Fix**: Implement uncertainty-aware training

5. **Error Type 5**: Keyword Bias
   - **Example**: "Great product but has some issues"
   - **Hypothesis**: Model relies too heavily on keyword polarity
   - **Potential Fix**: Implement aspect-based sentiment analysis
Some sentences contain positive-sounding words (“great,” “amazing”) alongside negatives. The model may rely too heavily on keyword polarity without enough context understanding. This explains why examples like ID 19 and ID 42 were misclassified.


### Limitations and Bias
- The model was trained on a relatively small dataset, which may limit its generalization to diverse text domains
- Performance may degrade with informal language, slang, or domain-specific terminology
- The model may inherit biases present in the training data


### Fairness & Robustness Observations

**Typos and Misspellings** :
Many examples in your dataset contain spelling errors (“latecny” instead of “latency”, “feles” instead of “feels”, “laetncy” instead of “latency”).
The model sometimes misclassifies these sentences because tokenizers treat misspelled words as unknown tokens or poorly contextualized embeddings.
- Observation: The model’s robustness decreases with noisy or misspelled inputs unless explicitly trained with noisy data augmentation.

**Emojis** :
Emojis (e.g., 😡, ✅, 🥴, 💯, 🚀) appear in several reviews.
The model handles emojis inconsistently — sometimes ignoring them or misinterpreting their sentiment.
- Observation: Emojis can carry strong sentiment signals (positive or negative), and ignoring them reduces the model’s accuracy for sentiment-heavy reviews.

**Code-switching / Mixed Language Content** :
A few examples contain mixed-language phrases or transliterations (“yaar”, “mast”, “fari”, “jugaad”).
The tokenizer and model were trained mostly on English data, so code-switching lowers accuracy.
- Observation: The model lacks robustness for code-switched text and requires multilingual fine-tuning or additional preprocessing to handle such cases effectively.

**Noisy Phrases & Repetition** :
Several reviews contain repeated keywords (“build build build”, “charging charging charging”), which add noise without extra meaning.
The model sometimes overweights repeated keywords, leading to incorrect sentiment prediction.
- Observation: This indicates a lack of robustness against noisy or redundant input, which could be addressed with preprocessing or noise-aware training.

**Domain-specific Terms** :
Several reviews contain technical or product-specific terms (“latency”, “firmware”, “pairing”), which may have sentiment implications depending on context.
The model sometimes misinterprets these terms if they are underrepresented in the training data, leading to incorrect sentiment classification.
- Observation: This indicates a lack of robustness to domain-specific vocabulary, which could be addressed with domain adaptation or fine-tuning on similar product review datasets.




