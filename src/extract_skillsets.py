import pandas as pd
import en_core_web_trf
import jsonlines
import PyPDF2
from spacy import displacy


nlp = en_core_web_trf.load()


def add_newruler_to_pipeline():
    nlp.add_pipe("entity_ruler", after='parser').from_disk('Data/skill_patterns.jsonl')


def visualize_entity_ruler(entity_list, doc):
    options = {"ents": entity_list}
    displacy.render(doc, style='ent', options=options)


def create_skill_set(doc):
    return set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])


def create_skillset_dict(resume_names, resume_texts):
    skillsets = [create_skill_set(resume_text) for resume_text in resume_texts]

    return dict(zip(resume_names, skillsets))


def extract_text_from_pdf(file):
    f_reader = PyPDF2.PdfFileReader(open(file, 'rb'))
    page_count = f_reader.getNumPages()
    text = [f_reader.getPage(i).extractText() for i in range(page_count)]
    return str(text).replace("\\n", "")


if __name__ == '__main__':
    add_newruler_to_pipeline()


"""add_newruler_to_pipeline()
txt = extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf')
doc = nlp(txt)
print([(ent.text, ent.label_) for ent in doc.ents])

print(create_skill_set(doc))"""
