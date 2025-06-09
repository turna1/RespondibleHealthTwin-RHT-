# RHealthTwin
> 🧭 *In October 2023, the [World Health Organization (WHO)](https://www.who.int/publications/i/item/9789240085355) published a formal call for action to ensure the ethical and responsible use of large language models (LLMs) in healthcare, urging developers to uphold principles such as transparency, fairness, safety, accountability, and human agency.*

In response to this call, we present **ResponsibleHealthTwin (RHT)**—a principled, modular framework for creating *WHO-aligned, multimodal digital twins* that support responsible LLM-based well-being guidance. At its core is the **Responsible Prompt Engine (RPE)**, which uses slot-based tagging and ethical instruction generation to ensure safe, explainable, and user-contextualized outputs from large language models.

![framework](https://github.com/user-attachments/assets/42712497-6e39-4fd1-88de-d988930e1db0)  ![examplescene](https://github.com/user-attachments/assets/42d75aa0-5f3e-4541-b686-269439f1f031)



**Responsible Health Twin (RHealthTwin)** is an open research framework for building ethically aligned, multimodal Digital Twins for personalized well-being applications using LLMs.
This repository supports the paper _"RHealthTwin: Towards Responsible and Multimodal Digital Twins for Personalized Well-being"_ (IEEE J-BHI, 2025). 
---

## 🔍 What's in this Repo?

This GitHub repo provides:

- ✅ **Synthetic test prompts** from 4 public datasets:
  - [MentalChat16k](https://github.com/ChiaPatricia/MentalChat16K_Main)
  - [MTS-Dialog v3](https://github.com/abachaa/MTS-Dialog)
  - [NutriBench v2](https://huggingface.co/datasets/dongx1997/NutriBench)
  - [SensorQA](https://github.com/benjamin-reichman/SensorQA)

- ⚙️ **Example implementation** of slot-based prompt generation using [LLaMA 4](https://ai.meta.com/llama/)
- 🧪 **Evaluation scripts and outputs**, including BLEU, ROUGE-L, BERTScore, FS, CAS, ICS, and WHO-aligned WRR
- 📦 **Generated assistant responses** for each prompt-role combination
- 📂 **Figures and result plots** used in the paper
- 🚧 Paper preprint will be uploaded soon

---

## 🌐 Live Demo (Hugging Face)

Test the Responsible Prompt Engine in action at:  
🔗 **[Hugging Face Demo Space](https://huggingface.co/spaces/Rahatara/WellebeingDT)**

This interactive UI lets you:
- Submit multimodal queries (text + screenshots)
- View dynamically generated structured prompts
- See model responses under ethical constraints
- Experiment with patient vs. provider prompt roles

---

## 🧠 Models Used

This repo evaluates and compares structured prompting across several general-purpose and biomedical LLMs:

- [GPT-4](https://openai.com/research/gpt-4)
- [Gemini 2.5 (Pro & Flash)](https://deepmind.google/technologies/gemini)
- [Meta LLaMA-4](https://ai.meta.com/llama/)
- [Qwen VL & Qwen2-7B](https://huggingface.co/Qwen)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [BioMistral-7B](https://huggingface.co/mistralai/BioMistral-7B)
- [Asclepius-7B](https://huggingface.co/starmpcc/Asclepius-7B)

Each model is tested with 4 prompting strategies: Zero-shot, Few-shot, Instruction-tuned, and our proposed RPE.

---

## 📖 Citation (Coming Soon)

We will update this section once the paper is officially published. Stay tuned!

---

## 📬 Contact

- Rahatara Ferdousi  
  [Email](mailto:rahatara.ferdousi@queensu.ca) | [Hugging Face](https://huggingface.co/Rahatara) | [LinkedIn](https://www.linkedin.com/in/rahatara)

---

> 💡 For any questions, feel free to open an issue or reach out!
