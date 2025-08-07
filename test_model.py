import unittest
from main import predict_match, process_text, correct_spelling, normalize_synonyms

class TestResumeJobMatching(unittest.TestCase):

    def test_basic_match(self):
        resume = "Experienced in Python, Django, and REST APIs."
        job = "Looking for someone with Django and Python experience building RESTful services."
        result = predict_match(resume, job)
        self.assertTrue(result['is_match'])
        print(f"Basic match score: {result['similarity_score']:.4f}")

    def test_typo_handling(self):
        resume = "Pyhton developer with experince in web devlopment"
        job = "Python developer experienced in web development"
        result = predict_match(resume, job)
        self.assertTrue(result['is_match'])
        print(f"Typo handling score: {result['similarity_score']:.4f}")

    def test_synonym_matching(self):
        resume = "Expert in JS and web frameworks"
        job = "Seeking developer with JavaScript experience"
        result = predict_match(resume, job)
        self.assertTrue(result['is_match'])
        print(f"Synonym match score: {result['similarity_score']:.4f}")

    def test_no_match(self):
        resume = "Certified accountant with 5 years in finance"
        job = "React developer with frontend experience"
        result = predict_match(resume, job)
        self.assertFalse(result['is_match'])
        print(f"No match score: {result['similarity_score']:.4f}")

    def test_process_text_pipeline(self):
        raw_text = "pyhton javascrip js"
        processed = process_text(raw_text)
        self.assertIn("python", processed)
        self.assertIn("javascript", processed)

    def test_normalize_synonyms(self):
        text = "js nlp"
        normalized = normalize_synonyms(text)
        self.assertIn("javascript", normalized)
        self.assertIn("natural language processing", normalized)

    def test_correct_spelling(self):
        text = "pyhton"
        corrected = correct_spelling(text)
        self.assertEqual(corrected.strip().lower(), "python")

if __name__ == "__main__":
    unittest.main()
