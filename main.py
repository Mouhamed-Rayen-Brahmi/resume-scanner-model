import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import sent_tokenize
import re
import os.path

MODEL_CONFIG = {
    'model_name': 'paraphrase-mpnet-base-v2',
    'spell_check': True,
    'use_synonyms': True,
    'chunk_long_text': True
}

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

model = SentenceTransformer(MODEL_CONFIG['model_name'])
spell = SpellChecker()


def load_synonyms_from_csv(csv_path='tech_synonyms.csv'):
    try:
        synonyms_df = pd.read_csv(csv_path)

        required_columns = ['synonym', 'canonical']
        if not all(col in synonyms_df.columns for col in required_columns):
            print(f"CSV must contain columns: {required_columns}")
            return {}

        synonyms_df = synonyms_df.dropna()
        synonyms_df['synonym'] = synonyms_df['synonym'].astype(str)
        synonyms_df['canonical'] = synonyms_df['canonical'].astype(str)

        synonyms_dict = dict(zip(synonyms_df['synonym'], synonyms_df['canonical']))

        print(f"Loaded {len(synonyms_dict)} technical term synonyms from {csv_path}")
        return synonyms_dict

    except Exception as e:
        print(f"Error loading synonyms CSV file: {e}")
        return {
            'pyhton': 'python',
            'javascrip': 'javascript',
            'js': 'javascript',
        }


tech_synonyms = load_synonyms_from_csv()


def correct_spelling(text):
    if not MODEL_CONFIG['spell_check']:
        return text

    words = text.split()
    corrected_words = []

    for word in words:
        if re.match(r'^[a-zA-Z]+$', word) and len(word) > 2:
            corrected = spell.correction(word.lower())
            if corrected:
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)


def normalize_synonyms(text):
    if not MODEL_CONFIG['use_synonyms']:
        return text

    words = text.lower().split()
    normalized_words = [tech_synonyms.get(word, word) for word in words]
    normalized_text = ' '.join(normalized_words)

    for key, value in tech_synonyms.items():
        if ' ' in key:
            normalized_text = normalized_text.replace(key, value)

    return normalized_text


def process_text(text):
    text = correct_spelling(text)
    text = normalize_synonyms(text)
    return text


def chunk_and_encode(text, model):
    if not MODEL_CONFIG['chunk_long_text'] or len(text.split()) <= 30:
        return model.encode([text])[0]

    sentences = sent_tokenize(text)

    if len(sentences) <= 1:
        return model.encode([text])[0]

    sentence_embeddings = model.encode(sentences)
    return np.mean(sentence_embeddings, axis=0)


def train_model_threshold(resume_csv='resume.csv', jobs_csv='job_title_des.csv'):
    try:
        if not (os.path.isfile(resume_csv) and os.path.isfile(jobs_csv)):
            print(f"Warning: One or both CSV files not found. Using default threshold.")
            return 0.6
        resumes_df = pd.read_csv(resume_csv)
        jobs_df = pd.read_csv(jobs_csv)

        print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} jobs")

        resume_text_col = 'skills' if 'skills' in resumes_df.columns else 'content'
        if resume_text_col not in resumes_df.columns:
            print(f"Could not find resume text column. Available columns: {resumes_df.columns.tolist()}")
            return 0.6

        similarities = []
        labels = []

        resume_samples = resumes_df.head(3)
        job_samples = jobs_df.head(3)

        for _, resume_row in resume_samples.iterrows():
            try:
                resume_text = str(resume_row[resume_text_col])

                for _, job_row in job_samples.iterrows():
                    try:
                        job_text = f"{job_row['Job Title']} {job_row['skills']} {job_row.get('skills', '')}"

                        processed_resume = process_text(resume_text)
                        processed_job = process_text(job_text)

                        resume_embedding = chunk_and_encode(processed_resume, model)
                        job_embedding = chunk_and_encode(processed_job, model)

                        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

                        is_match = similarity > 0.6

                        similarities.append(similarity)
                        labels.append(1 if is_match else 0)
                    except Exception as e:
                        print(f"Error processing job: {e}")
                        continue
            except Exception as e:
                print(f"Error processing resume: {e}")
                continue

        if similarities:
            print("similarities:", similarities)
            return np.mean(similarities)
        else:
            print("No similarities found. Using default threshold.")
            return 0.6

    except Exception as e:
        print(f"Error training model: {e}")
        return 0.6


best_threshold = train_model_threshold()
print(f"Best threshold determined: {best_threshold:.2f}")


def predict_match(resume_text, job_description):
    processed_resume = process_text(resume_text)
    processed_job = process_text(job_description)

    resume_embedding = chunk_and_encode(processed_resume, model)
    job_embedding = chunk_and_encode(processed_job, model)

    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

    threshold = max(best_threshold, 0.20)

    is_match = similarity > threshold

    return {
        'similarity_score': similarity,
        'is_match': bool(is_match),
        'processed_resume': processed_resume,
        'processed_job': processed_job
    }
