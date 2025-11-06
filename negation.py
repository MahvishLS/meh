import pandas as pd

df = pd.read_csv('clinical_sentences.csv')

keywords = {
    "pneumonia": ["pneumonia", "lung infection"],
    "diabetes": ["diabetes", "hyperglycemia", "diabetic"],
    "hypertension": ["hypertension", "high blood pressure"],
    "tuberculosis": ["tuberculosis", "TB"],
    "myocardial_infarction": ["myocardial infarction", "heart attack"],
    "asthma": ["asthma", "wheeze"],
    "hepatic_steatosis": ["hepatic steatosis", "fatty liver"],
    "eczema": ["eczema", "dermatitis"],
    "osteoporosis": ["osteoporosis", "bone loss"],
    "migraine": ["migraine", "headache"],
    "COPD": ["COPD", "chronic obstructive pulmonary disease"]
}

import re

neg_terms = r"\b(no|not|absence of|negative for|rules out|no evidence of|free of)\b"

def detect(text, cond):
  text = text.lower()
  found = False
  for k in keywords[cond]:
    for m in re.finditer(re.escape(k), text):
      found = True
      if not any(
          m.start() - n.end() <= 40 and n.end() < m.start()
          for n in re.finditer(neg_terms, text)
      ):
        return "Present"

  return "Absent" if found else None

out = []
for s in df['sentence']:
  row = {'Sentence': s}
  for cond in keywords:
    row[cond] = detect(s, cond)
  out.append(row)

pd.DataFrame(out).head(3)
