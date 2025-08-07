# 🧠 Resume & Job Description Matcher

This project intelligently matches **resumes** with **job descriptions** using **semantic embeddings**, **spelling correction**, and **technical synonym normalization**.

> "Find the best candidate faster — even if they wrote *'pyhton'* instead of *'python'*."

---

## 🚀 Features

- ✅ **Spelling Correction** (e.g., *pyhton → python*)
- ✅ **Tech Synonym Normalization** (e.g., *JS → JavaScript*)
- ✅ **Sentence Embeddings** with [`paraphrase-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- ✅ **Automatic Similarity Threshold Training** using real resumes and job descriptions
- ✅ **Chunking for Long Texts**
- ✅ **Unit Tests** included

---

## 📦 Dependencies

Install with pip:

```bash
pip install -r requirements.txt
