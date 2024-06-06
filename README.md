# Supportiv MLE Intern Medical Assistant Assignment

### Problem Statement

Develop a medical question-answering system utilizing the provided dataset containing medical information. The goal is to create a model that can effectively answer user queries related to medical diseases. You are free to augment the provided dataset with other medical datasets if you feel it is necessary. 

### Approach and Assumptions

1. Upon initially reviewing the task of developing a medical chatbot, my immediate idea was to leverage Large Language Models (LLMs). However, as the end of the task specifies no third-party LLM APIs (such as OpenAI, Claude), I focused my attention on using open-source LLMs that could be fine-tuned and run locally on my machine or on Google Colab. I have assumed the integrity of the question answer pairs provided in the dataset and that they provide adequate coverage of medical questions that the model will be expected to answer.
 
2. Initially, I decided to fine-tune a GPT-2 model for the medical question-answering task.I preprocessed the dataset by removing NaN values, eliminating duplicate questions, and saving it to a text file ('input.txt') which was fed into GPT-2 model training.However, the fine-tuned GPT-2 model was not very accurate and struggled with hallucinations, misunderstanding the context of questions, and providing incorrect outputs. Additionally, training it on a simple file of Q&A pairs without a  specific **context column** did not feel like the **best approach for this task**.

3. I ultimately decided to use the `mlx-community/quantized-gemma-2b` model because I could leverage Apple's MLX library (A framework for machine learning research on Apple silicon chips) to fine-tune it locally on my MacBook Pro using MLX-LoRA on the medical dataset provided.
Gemma is a family of lightweight, state-of-the-art open models built on the research and technology used to create the Gemini models. Gemma-2b's quantization reduces the memory footprint and increases the inference speed, enabling deployment and training on devices with limited resources. Despite its smaller size, the Gemma-2b model demonstrates performance comparable to larger models, as shown by the MMLU benchmark comparison with other models.
<img width="1186" alt="MMLU Benchmarks for Gemma compared to other models." src="https://github.com/Arjun-254/SupportivMLE-InternFinetune/assets/102243820/898b44d5-1647-4e64-9564-7e61fa787690">

### 1) Data Preprocessing
To format the question and answer columns to match the inputs Gemma was trained on, each question-answer pair in the dataset is transformed into a specific structured format. This format includes special tokens like <bos>, <start_of_turn>, <end_of_turn>, and <eos> to clearly differentiate between the user's question and the model's response. After formatting, I split into training and validation sets, with 90% allocated to training and 10% to validation. I then shuffled the data for randomness before being saved as JSON Lines in the data folder for the finetuning step.

Example input - ` {"text": "<bos><start_of_turn>user\nWhat are the symptoms of Glaucoma?<end_of_turn>\n<start_of_turn>model\nGlaucoma symptoms include blurred vision, eye pain, headache and nausea.<end_of_turn><eos>"} `

### 2) Model Training
LoRA allows efficient fine-tuning of Gemma-2b for our open-source medical chatbot. Instead of updating the entire model's massive number of parameters, LoRA trains a small set of additional weights that are inserted into the pre-trained Gemma Model.This approach significantly reduces the computational requirements and memory footprint during fine-tuning.

Initially I tried to finetune the model using a batch size equal to 4 and 16 lora-layers, However this led to `Insufficient Memory` errors and I had reduce memory use for LoRA fine-tuning which had adequate resources in the documentation [Link to MLX Memory Isssues docs - https://github.com/ml-explore/mlx-examples/tree/main/lora#Memory-Issues]
#### Final LoRA Training Parameters
- **Model:** `mlx-community/quantized-gemma-2b`
- **Iterations:** 600
- **Batch Size:** 1
- **LoRA Layers:** 4
- **Learning Rate:** 1.000e-05

### 3) Model Evaluation 
- training loss (loss) is used to guide the optimization process during training, ensuring the model fits the training data well
- validation loss (val_loss) is crucial for assessing the model's ability to generalize to new data and helps in preventing overfitting

From the validation versus loss curve below, we can observe that the model did not fully converge, as evident from the fluctuations in both the training and validation loss values. The table also substantiates this observation as the given datapoints show that the loss values have not stabilized/reached a steady state. 


| Iteration   | Training Loss       | Validation Loss     |
|-------------|---------------------|---------------------|
| 1           | 2.416               | 2.247               |
| 200         | 1.913               | 1.813               |
| 400         | 1.923               | 1.708               |
| 600         | 1.547               | 1.637               |

![image](https://github.com/Arjun-254/SupportivMLE-InternFinetune/assets/102243820/1224c61f-3a15-4b9c-931f-8c408a64cbc6)

Training large models such as Gemma often demands significant computational resources, particularly in terms of memory and GPU capabilities.
Further training iterations were not possible due to the limitations of the available hardware due to the known Memory Issues encountered while finetuning these large models locally using MLX.

Additionally,I have created a model card with a usage guide, and uploaded all the files to my Hugging Face account. It can be found at [https://huggingface.co/Arjun24420/Medical-quantized-gemma-2b].

### 4) Sample Model Outputs

**(Q1)** What is Glaucoma ?
 - Glaucoma is a common eye condition that can cause vision loss or blindness if not treated. Glaucoma occurs when the
optic nerve is damaged due to increased eye pressure. It is usually caused by a narrowing of the drainage canals in the
eye. This condition is caused by an increase in eye pressure called intraocular pressure (IOP). Intraocular pressure is
caused by several factors including aging, race, gender, lifestyle habits, and medical conditions. Intraocular pressure
can increase due to aging, race, gender,

**(Q2)** What is Leukemia ?
 -   Leukemia is a type of cancer that starts in the blood cells. The cancer cells form in the bone marrow, which is the
soft tissue inside the bones where the blood cells are made. When the cancer cells form, they crowd out the normal blood
cells, which makes the person more likely to get infections and have other health problems.  What are the symptoms of
Leukemia ? Leukemia is usually diagnosed based on symptoms and a physical examination. The symptoms of leukemia include
fatigue, easy bruising, bleeding, and frequent

**(Q3)** What is Diabetic Retinopathy ?
 -   Diabetic retinopathy is a condition that occurs when blood vessels in the retina of the eye are damaged by diabetes.
Diabetic retinopathy is a common complication of diabetes. It is caused by the high levels of glucose in the blood that
causes damage to the tiny blood vessels in the retina. Diabetic retinopathy can lead to vision loss if it is not
treated. Diabetic retinopathy can affect people with type 1 or type 2 diabetes. Diabetic retinopathy usually occurs in
people with type 2 diabetes who have

**(Q4)** What are the symptoms of depression?
 -   Depression is a common mental health condition that affects people of all ages and genders. It can be caused by a
variety of factors, including genetics, brain chemistry, and life stressors. Symptoms of depression can include feelings
of sadness, hopelessness, and emptiness; changes in sleep patterns; changes in eating habits; changes in energy levels;
and feelings of guilt or inadequacy. Depression can also cause changes in behavior, including decreased motivation and
interest in activities that were once enjoyable, decreased energy levels, and changes in



### 5) Potential improvements
- One of my initial assumptions was the integrity and adequate question coverage of the dataset. A future improvement is to supplement the dataset with additional medical question-answer pairs from other relevant and high-quality medical datasets leading to a greater coverage of medical questions.
- Another improvement would be in the form of hardware support to allow the finetuning of the gemma-2b model till weight convergence. In the current state my laptop's hardware constraints have limited the training iterations,batch size and LoRA-layers impacting the models ability to fully converge. A powerful GPU and increased RAM would mitigate this limitation. Additionally this would also allow me to fine-tune the larger Gemma-7b model which a part of the same model family. This model will likely yield more accurate results due to its larger size.

All of these improvements would improve the model's ability to generalize and answer a wider range of medical questions accurately.
