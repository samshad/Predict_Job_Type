import pandas as pd
import handler as h


df = pd.read_csv('../Data/Test/jobs.csv')
df = df[['jobTitle', 'cleanContent', 'jobSector', 'uid']]
df.rename(columns={'jobTitle': 'title', 'cleanContent': 'job_description', 'jobSector': 'category',
                   'uid': 'judi'}, inplace=True)
#print(df.head().to_string(index=False))
#df.to_csv('Data/jobs.csv', index=False)
x = []
for i in df['job_description']:
    t = h.cleaner(str(i) + str(df['title']))
    x.append(t)
    print(t)

df['clean_description'] = pd.Series(x)
df.to_csv('Data/jobs.csv', index=False)
