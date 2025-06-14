import pandas as pd

columns = ['person_couple', 'conversation', 'explaination', 'toxic']

df1 = pd.read_csv('./all_data/classification_and_explaination_toxic_conversation.csv')
df1 = df1[columns]

print (df1['person_couple'].value_counts())
print (df1['toxic'].value_counts(), '\n\n\n')



df2 = pd.read_csv('./all_data/explaination_toxic_conversation_most_toxic_sentences.csv')
df2 = df2[['person_couple', 'conversation', 'explaination']]

print (df2['person_couple'].value_counts(), '\n\n\n')



df3 = pd.read_csv('./all_data/explaination_toxic_conversation.csv')
df3 = df3[columns]

print (df3['person_couple'].value_counts())
print (df3['toxic'].value_counts())