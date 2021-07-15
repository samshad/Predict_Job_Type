import pandas as pd
import predict_job_class as predict_category
import extract_skillsets as es
import text_cleaner as tc
es.add_newruler_to_pipeline()


def get_skillsets(doc):
    doc = es.nlp(doc)
    #print([(ent.text, ent.label_) for ent in doc.ents])
    return es.create_skill_set(doc)


def match_skills(vacature_set, cv_set, resume_name):
    if len(vacature_set) < 1:
        print('could not extract skills from job offer text')
    else:
        pct_match = round(len(vacature_set.intersection(cv_set)) / len(vacature_set) * 100, 0)
        if pct_match >= 40:
            print(resume_name + " has a {}% skill match on this job offer".format(pct_match))
            print('Asked skills: {} '.format(vacature_set))
            print('Matched skills: {} \n'.format(vacature_set.intersection(cv_set)))

        #return (resume_name, pct_match)


jobs_df = pd.read_csv('Data/train_jobs.csv')

"""resume_txt = es.extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf')
resume_class = predict_category.get_job_class(resume_txt)[0]
print(f"Resume type: {resume_class}")
resume_skills = get_skillsets(resume_txt)
print(f"Resume Skills: {resume_skills}")"""

df = pd.read_csv('Data/Resumes/Archive/ResumeDataSet_1.csv')
# print(df['category'].value_counts())
tf = df[df['category'] == 'HR']
# tf = df[df['category'] == 'Java Developer']
# tf = df[df['category'] == 'Mechanical Engineer']
# tf = df[df['category'] == 'Business Analyst']
# print(tf['category'].value_counts())

cnt = 0
for index, row in tf.iterrows():
    resume_txt = tc.cleaner(row['resume'])
    resume_class = predict_category.get_job_class(resume_txt)[0]
    print(f"Resume type: {resume_class}")
    resume_skills = get_skillsets(resume_txt)
    print(f"Resume Skills: {resume_skills}")

    specific_jobs = jobs_df[jobs_df['category'] == resume_class]
    print(specific_jobs['category'].value_counts())

    for index, row in specific_jobs.iterrows():
        job_skills = get_skillsets(row['clean_description'])
        match_skills(job_skills, resume_skills, 'Mr. X')

    if cnt >= 2:
        break
