import numpy as np
import pandas as pd
import pyreadstat
import streamlit as st
import seaborn as sns
import auxiliary_functions as af
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import chi2_contingency


st.set_page_config(layout="wide")
st.title('Machine and Deep learning techinques to predict the severity of symptoms in Covid patients')
st.subheader('Programming project')
st.subheader('Jaume Betriu, Verona February 2022')
dades=pd.read_spss(r'C:\Users\Jaume\Desktop\Exactes\Projecte_Prog\EarlyLifeCovidPACIENTES V6 ID.sav')

st.write('Our database consists on data about patients from the Hospital Clínic de Barcelona and our objective will be to build a machine learning model capable of predicting the severity of symtoms of the disease in patients')
st.write('The original Data Base looks like this:')
if st.checkbox('Show original data'):
  st.write(dades.head(10))

st.write('The first thing we will do is translate to english the data. The translated data set is:')

dades['Tabaco_SIno']=dades['Tabaco_SIno'].apply(af.canvi_tab)
dades['UCI']=dades['UCI'].apply(af.canvi_UCI)
for columna in dades.columns[14:]:
  dades[columna]=dades[columna].apply(af.canvi)
dades['Gender']=dades['Gender'].apply(af.canvi_sex)
dades['IUGR_calc']=dades['IUGR_calc'].apply(af.canvi_UCI)
dades.columns=pd.Index(['ID', 'UCI', 'IUGR_missing', 'Age', 'Gender', 'Size', 'Weight',
       'Size_mt', 'IMC', 'BW', 'BW_2500', 'percentil_birth', 'IUGR_calc',
       'Tobacco_yes_no', 'Hipertension', 'Heart_diseases', 'DM', 'Dyslipidemia',
       'Obesity', 'Kidney_disease', 'Autoimmune', 'Cancer', 'Thyroid', 'Infectious',
       'Psychiatric'])

if st.checkbox('Show translated data'):
  st.write(dades.head(10))

if st.checkbox('Information about the columns'):
  st.header('Brief explanation of the variables')
  st.write('If we look at the columns we will see that we have information about:')
  st.write('1. **ID**: Identification of the patient')
  st.write('2. **UCI**: If the patient ended in the Unitat de Cures Intensives, Intensive Care Unit in english')
  st.write('3. **IUGR_missing**: We will not use this column beacuse it does not have any information')
  st.write('4. **Age**')
  st.write('5. **Gender**')
  st.write('6. **Size**: Size of the patient')
  st.write('7. **Weight**')
  st.write('8. **Size_mt**: Size divided by 10')
  st.write('9. **IMC**: Body mass index')
  st.write('10. **BW**: Birth weight of the patient')
  st.write('11. **BW_2500**: If the birth weight of the patient was normal or not')
  st.write('12. **percentil_birth**')
  st.write('13. **IUGR_calc**: Intrauterina Growth Restriction')
  st.write('14. **Tobacco_yes_no**: Smoking habit present during the life of the patient?')
  st.write('15.-25. Has the patient suffered any disease related to the name of the column?')
  st.write('17. **DM**: Diabetes Mellitus')

  st.header('Information about null values:')
  null_values_info=pd.DataFrame(dades.isna().sum())
  null_values_info.columns=['Number of null values']
  st.write(null_values_info)
  st.write('* The dataset has no null values')

st.header('Initial exploration of the data:')
st.write('First we want to know how the variables are distributed and extract some information from them:')
col_1, col_2=st.columns(2)
with col_1:
  st.write('Histogram of the variable **Age**:')
  fig, ax=plt.subplots(figsize=(10,6))
  plt.hist(dades['Age'],edgecolor='black',linewidth=1.2)
  st.write(fig)
with col_2:
  st.write('Histogram of the variable **Weight**:')
  fig, ax=plt.subplots(figsize=(10,6))
  plt.hist(dades['Weight'],color='orange',edgecolor='black',linewidth=1.2)
  st.write(fig)

st.write('* We can see that the data is pretty well distributed between the ages of 25 and 65. This means that we have a good statistical sample regarding the age.')
st.write('* With the Weight we have a mean arround 70  witch makes sense. It will be better to divide between mean and woman to get better information')
  
col_3, col_4=st.columns(2)
with col_3:
  st.write('Boxplot of the variable **Weight** depending on the **Gender**:')
  fig, ax=plt.subplots(figsize=(10,6))
  sns.boxplot(x='Gender',y='Weight',data=dades,palette=sns.color_palette('Set2'))
  st.write(fig)

with col_4:
  st.write('Boxplot of the variable **IMC** depending on the **Gender**:')
  fig, ax=plt.subplots(figsize=(10,6))
  sns.boxplot(x='Gender',y='IMC',data=dades,palette=sns.color_palette())
  st.write(fig)

st.write('* As we expected we see an increase on the **weight** of men')
st.write('* The mean of **weight** in men is arround 82 kg wich is a little higher than the Spanish mean that is arround 75.8 kg. This increase could be related with the fact that our data tends to have more individuals in the age ranges of 45 to 65')
st.write('* Exactly the same situation happens with women')

st.title('Study of the weight of the patients in relation with complications during pregnancy')
st.write('It might be interesting to explore the boxplot of the **actual weight of the patient** grouped by their **birth weight** and **IUGR** to see if patients tend to weight less during their lives if they had complications in their development during pregnancy')
fig, ax=plt.subplots(figsize=(10,6))
sns.boxplot(y='Weight',x='BW_2500',data=dades,hue='IUGR_calc',palette=sns.color_palette("Paired"))
st.write(fig)
st.write('* We can clearly  see that those patients who had a **normal birth weight** and no **IUGR** tend to have a higher weight')
st.write('* In the case of patients with abnormal **BW** we can see that suffering **IUGR** does not have significant impact in the future **weight**')
st.write('* As a general conclusion **both pregnancy complications lead to a decrease of the weight of the patient during the adulthood**')

st.header('Relation of the categorical variables of the data set with **UCI**')

llista_dades=[]
column_names=list(dades.columns[13:])+['Gender','BW_2500','IUGR_calc']
for columna in column_names:
  llista_dades.append(pd.crosstab(index=dades[columna],columns=dades['UCI'],normalize='index'))

llista_dades1=pd.Series(llista_dades).values.reshape(5,3)

fig,ax=plt.subplots(5,3,figsize=(17, 22))
for i in range(llista_dades1.shape[0]):
  for j in range(llista_dades1.shape[1]):
    llista_dades1[i,j].plot(ax=ax[i,j],kind='bar',stacked='True')
    ax[i,j].legend(title='UCI',loc='lower left')

st.write(fig)
st.write('Looking at this plot we can highlight:')
st.write('1. **Tobacco consumption** has an impact on the probability of ending in the *UCI* in the case that you get COVID but is not as high as we might have expected')
st.write('2. **Heart diseases, Diabetes, Dyslipidemia, Cancer, Infectious** and **Intrauterina Growth Restriction** seem to have an impact on the probability of ending in the UCI')
st.write('3. **Hipertension** and **Kidney disease** have a huge negative inpact in the probability of ending up in the UCI')
st.write('4. **Obesity, Autoimune diseases, Tyroid diseases** and **Psychiatric diseases** seem to have little or no relation with the probability of suffering severe symptoms')
st.write('5. **Gender** has a huge impact, beeing men who have the highest probability')
st.write('6. **BW_2500** has a big impact too, meaning that **if your body weight when you were born was lower than normal (2500 grams) the probability of ending at the UCI will be higher**')

st.header('Correlation between continuous variables:')
dades.drop('IUGR_missing',axis=1)
correlation=dades.loc[:,'Age':].corr()
mask = np.triu(np.ones_like(correlation, dtype=np.bool))
for i in range(mask.shape[0]):
  mask[i,i]=False

fig, ax=plt.subplots(figsize=(7,4))
sns.heatmap(correlation,mask=mask,annot=True)
st.write(fig)
st.write('We can highlight some correlations that we already expected as :')
st.write('1. **Weight-Size**')
st.write('2. **Weight-IMC**')
st.write('3. **Percentile birth-Birth weight**')

st.write('This shows that the **percentile_birth** has been calculated using data from a bigger dataset')

st.header('Correlation between categorical variables')

dades_categoriques=dades[['UCI']+['BW_2500']+list(dades.columns[list(dades.columns).index('IUGR_calc'):len(dades.columns)])].apply(lambda x : pd.factorize(x)[0])+1
from scipy.stats import chi2_contingency
factors_paired = [(i,j) for i in dades_categoriques.columns.values for j in dades_categoriques.columns.values] 

chi2, p_values =[], []

for f in factors_paired:
    if f[0] != f[1]:
        chitest = chi2_contingency(pd.crosstab(dades_categoriques[f[0]], dades_categoriques[f[1]]))   
        chi2.append(chitest[0])
        p_values.append(chitest[1])
    else:      # for same factor pair
        chi2.append(0)
        p_values.append(0)

chi2 = np.array(chi2).reshape((15,15)) # shape it as a matrix
p_values=np.array(p_values).reshape((15,15))
chi2 = pd.DataFrame(chi2, index=dades_categoriques.columns.values, columns=dades_categoriques.columns.values) # then a df for convenience
p_values=pd.DataFrame(p_values, index=dades_categoriques.columns.values, columns=dades_categoriques.columns.values)



if st.checkbox('See here the heatmap of the p-values'):
  mask = np.triu(np.ones_like(p_values, dtype=np.bool))
  for i in range(mask.shape[0]):
    mask[i,i]=False
    fig, ax=plt.subplots(figsize=(15,7))
    sns.heatmap(p_values, mask=mask,annot=True)
  st.write(fig)

mask = np.triu(np.ones_like(chi2, dtype=np.bool))
for i in range(mask.shape[0]):
  mask[i,i]=False

fig,ax=plt.subplots(figsize=(20,12)) 
sns.heatmap(chi2, mask=mask,annot=True)
st.write(fig)

st.write('Looking at the heatmap and the p-values heatmap we can get important information about relations between some diseases')
st.write('1. **IUGR** and **BW_2500** have a huge correlation. Intrauterine growth restriction leads frequently to anormal birth weight')
st.write('2. **Hipertension** is related to **Kidney disease, Tyroids, Dyslipidemia, Diabetes** and **Heart diseases**')
st.write('3. **Diabetes** is related with **Dyslipidemia** and **Obesity** as we might have expected')
st.write('4. **Tobacco** is related obiously to developing some kinf of **cancer**')
st.write('5. **Hipertension** diseases have a high correlation with the **UCI** variable')

st.header('Machine learning modeling of the data')








st.header('Deep learning model of the data using Neural Networks(nn)')