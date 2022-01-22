import numpy as np
import pandas as pd
import pyreadstat
import streamlit as st
import seaborn as sns
import auxiliary_functions as af
import matplotlib.pyplot as plt
import io

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

if st.checkbox('Information about the columns:'):
  st.header('Brief explanation of the variables:')
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
  st.write('The dataset has no null values')

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

