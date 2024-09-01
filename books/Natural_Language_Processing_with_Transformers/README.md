# Natural Language Processing with Transformers


### Chapter 2. Text classification

There are 2 ways of training a text classifier
- Feature extraction
- Fine-tuning

**Training text classifier using feature extraction**
Using a transformer as a feature extractor is fairly simple. As shown in Figure 2-5, we freeze the body’s weights during training and use the hidden states as features for the classifier. The advantage of this approach is that we can quickly train a small or shallow model. Such a model could be a neural classification layer or a method that does not rely on gradients, such as a random forest. This method is especially convenient if GPUs are unavailable, since the hidden states only need to be precomputed once.

Key steps:
1. Loading the Model and Tokenizer:
We load the DistilBERT model and tokenizer using AutoModel and AutoTokenizer. The model is placed on the GPU if available.

2. Tokenizing the Dataset:
The dataset is tokenized with padding and truncation enabled. This converts text into token IDs and attention masks.

3. Extracting Hidden States:
We define a function extract_hidden_states that processes each batch of inputs, passes them through the model, and retrieves the last hidden state for the [CLS] token (a summary of the input sequence).

4. Mapping the Extraction Function:
The map() function is used to apply the hidden state extraction across the entire dataset, creating a new hidden_state column.

5. Creating Feature Matrices:
The hidden states are converted into NumPy arrays to be used as input features for a classifier.

6. Training a Classifier:
We use a simple Random Forest classifier from Scikit-learn to train on the extracted features and labels.

7. Evaluating the Model:

The classifier is evaluated using the validation set, and a classification report is generated.


**Training text classifier using fine tuning**

The first thing we need is a pretrained DistilBERT model like the one we used in the feature-based approach. The only slight modification is that we use the AutoModelFor
SequenceClassification model instead of AutoModel. The difference is that the AutoModelForSequenceClassification model has a classification head on top of the
pretrained model outputs, which can be easily trained with the base model. We just need to specify how many labels the model has to predict.
```
from transformers import AutoModelForSequenceClassification
num_labels = 6
model = (AutoModelForSequenceClassification
.from_pretrained(model_ckpt, num_labels=num_labels)
.to(device))
```

To train, we just need to specify the hyperparameters, Hugging face will do everything for you
```
training_args = TrainingArguments(output_dir=model_name,
num_train_epochs=2,
learning_rate=2e-5,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
weight_decay=0.01,
evaluation_strategy="epoch",
disable_tqdm=False,
logging_steps=logging_steps,
push_to_hub=True,
log_level="error")
```

**Error analysis**
Here’s a summary of the main ideas behind the error analysis process described:

1. **Objective**: The goal is to analyze your model’s performance by inspecting the examples where it struggles (high loss) and where it is most confident (low loss). This helps you identify issues like mislabeled data, dataset quirks, or potential biases in the model.

2. **Calculating Loss for Each Sample**:
   - During evaluation, the loss is calculated for each sample in the validation set. Higher loss values indicate predictions that are far from the true label, highlighting challenging examples for the model.

3. **Identifying Problematic Samples**:
   - Sorting the validation samples by loss allows you to easily identify examples where the model’s predictions are poor.
   - These samples often include wrongly labeled data or texts that are inherently difficult to classify due to ambiguity, special characters, or complex phrasing.

4. **Evaluating Model Confidence**:
   - Inspecting samples with low loss (where the model is very confident) helps ensure that the model isn’t relying on irrelevant patterns or shortcuts. Consistent low-loss predictions for certain classes might indicate overconfidence or dataset bias.

5. **Dataset Improvement**:
   - By detecting mislabeled data or ambiguous cases, you can refine your dataset, leading to significant performance gains without needing a larger model or more data.

6. **Conclusion**: Error analysis provides actionable insights for improving model performance by focusing on problematic examples and understanding where the model is struggling or overly confident. This process is a crucial step before fine-tuning or deploying the model.

This approach is a systematic way to improve your model’s accuracy and robustness by addressing underlying issues in the dataset or labeling.


**Saving and sharing the model**

The NLP community benefits greatly from sharing pretrained and fine-tuned models, and everybody can share their models with others via the Hugging Face Hub. Any
community-generated model can be downloaded from the Hub just like we downloaded the DistilBERT model. With the Trainer API, saving and sharing a model is
simple:
```
trainer.push_to_hub(commit_message="Training completed!")
```

### Chapter 7. Question Answering

[on the page 188, putting it all together] 