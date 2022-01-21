import numpy as np
import pandas as pd
import pyreadstat
import streamlit as st
import auxiliary_functions as af

st.title('Machine and Deep learning techinques to predict the severity of symptoms in Covid patients')
st.subheader('Programming project')
st.subheader('Jaume Betriu, Verona February 2022')
dades=pd.read_spss(r'C:\Users\Jaume\Desktop\Exactes\Projecte_Prog_Hospital_clínic\EarlyLifeCovidPACIENTES V6 ID.sav')

st.write('Our database consists on data about patients from the Hospital Clínic de Barcelona and our objective will be to build a machine learning model capable of predicting the severity of symtoms of the disease in patients')
st.write('The Data Base looks like this:')

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
st.write(dades.head(10))

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

st.write('coses escrites')

