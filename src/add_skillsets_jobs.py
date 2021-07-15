import pandas as pd
import extract_skillsets as es
es.add_newruler_to_pipeline()


df = pd.read_csv('Data/train_jobs.csv')
tf = df[df['category'] == 'IT/Telecommunication'].copy()
arr = []
for txt in tf['clean_description'][:5]:
    doc = es.nlp(txt)
    print([(ent.text, ent.label_) for ent in doc.ents])
    skills = es.create_skill_set(doc)
    print(skills)
    arr.append([skills])

tf['skills'] = pd.Series(arr)
print(tf.head().to_string(index=False))

"""txt = es.extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf')
es.add_newruler_to_pipeline()
doc = es.nlp(txt)
print([(ent.text, ent.label_) for ent in doc.ents])

print(es.create_skill_set(doc))"""
