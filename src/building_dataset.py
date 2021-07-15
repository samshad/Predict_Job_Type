import pandas as pd
import text_cleaner as tc


"""
df = pd.read_csv(r'E:\Work\Predict_Job_Type\Data\Test\train.csv')
df = df[['category', 'job_description']]
'''df.rename(columns={'jobTitle': 'title', 'cleanContent': 'job_description', 'jobSector': 'category',
                   'uid': 'judi'}, inplace=True)'''
#print(df.head().to_string(index=False))
#df.to_csv('Data/jobs.csv', index=False)
x = []
for i in df['job_description']:
    t = tc.cleaner(str(i))
    x.append(t)
    print(t)

df['clean_description'] = pd.Series(x)
df.to_csv('../Data/Test/train_cleaned.csv', index=False)

df = pd.read_csv('../Data/Test/test.csv')
x = []
for i in df['job_description']:
    t = tc.cleaner(str(i) + str(df['title']))
    x.append(t)
    print(t)
df['clean_description'] = pd.Series(x)
df.to_csv('../Data/Test/test_cleaned.csv', index=False)"""

"""
Garments/Textile                        656
Marketing/Sales                         633
NGO/Development                         526
IT/Telecommunication                    426
Engineer/Architect                      378
Accounting/Finance                      368
Medical/Pharma                          201
Commercial/Supply Chain                 175
HR/Org. Development                     143
"""

df = pd.read_csv(r'src/Data/translated_file.csv')
tf = df.drop((df[df['category'] != 'Garments/Textile'].index) & (df[df['category'] != 'Marketing/Sales'].index)
             & (df[df['category'] != 'NGO/Development'].index) & (df[df['category'] != 'IT/Telecommunication'].index)
             & (df[df['category'] != 'Engineer/Architect'].index) & (df[df['category'] != 'Accounting/Finance'].index)
             & (df[df['category'] != 'Medical/Pharma'].index) & (df[df['category'] != 'Commercial/Supply Chain'].index)
             & (df[df['category'] != 'HR/Org. Development'].index))
print(tf['category'].value_counts())
tf.to_csv(r'src/Data/train_jobs.csv', index=False)

