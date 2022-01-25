import numpy as np
import pandas as pd
import pyreadstat
import streamlit as st
import seaborn as sns
import auxiliary_functions as af
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC

primaryColor="#F63366"

st.title('Machine and Deep learning techinques to predict the severity of symptoms in Covid patients')
st.subheader('Programming project')
st.subheader('Jaume Betriu, Verona February 2022')
dades=pd.read_spss(r'C:\Users\Jaume\Desktop\Exactes\Projecte_Prog\EarlyLifeCovidPACIENTES V6 ID.sav')

st.write('Our database consists on data about patients from the Hospital Clínic de Barcelona and our objective will be to build a machine learning model capable of predicting the severity of symtoms of the disease in patients')
st.write('The original Data Base looks like this:')
with st.expander('Show original data'):
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

with st.expander('Show translated data'):
  st.write(dades.head(10))

with st.expander('Information about the columns'):
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



with st.expander('See here the heatmap of the p-values'):
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
st.write('In this part of the project our objective is to use machine learning techinques to predict the severity of symtoms. We will divide our data in a train sample (80 %) and a test sample (20 %) to test if the model works well with unseen data')

st.code('''dades_train = dades.sample(frac=0.8, random_state=25).drop('ID',axis=1)
dades_test = dades.drop(dades_train.index).drop('ID',axis=1)''')
dades_train = dades.sample(frac=0.8, random_state=25).drop('ID',axis=1)
dades_test = dades.drop(dades_train.index).drop('ID',axis=1)
st.write('The first model we will train is a logistic regression with all the predictors and we get the information:')

variables='+'.join(dades_train.columns.difference(['UCI','BW','IUGR_missing']))
formula='UCI~'+variables
model_log=smf.glm(formula=formula,data=dades_train,family=sm.families.Binomial()).fit()

with st.expander('Check the summary of the model: Logistic regression with the raw data'):
  st.write(model_log.summary())
st.write(' and trying different margins for the probability decision we get this precisions:')

with st.expander('Check the different precisions'):
  prediccions_prob=model_log.predict(dades_test)
  nombres=[]
  for i in range(1,20):
    nombres.append(i/20)
  for i in nombres:
    prediccions=['No' if x>i else 'Yes' for x in prediccions_prob]
    st.write('Margin of decission '+'**'+str(i)+'**')
    pr_yes, pr_no, pr_tot=af.precisions(dades_test['UCI'],prediccions)
    st.write(pr_yes)
    st.write(pr_no)
    st.write(pr_tot)
    st.write('________________________________________________')

st.write('We can see that we get high precisions in the case of the patients that did not suffer big complications during the disease. Unfortunately we get bad results in the accuracy predicting the cases that the patient will end up in the UCI wich is the thing that interests us. The best results that we get with this model are for the margin 0.8')

col_5,col_6=st.columns(2)
with col_5:
  st.write('This might be caused because the data that we are exploring is unbalanced:')
  st.write('Looking at the graphic we can see that we have arround 5 times more patients with no complications that with complications')
  st.write('')
with col_6:
  fig, ax=plt.subplots(figsize=(8,3))
  plt.bar(['No','Si'],[len(dades_train[dades_train['UCI']=='No']),len(dades_train[dades_train['UCI']=='Yes'])],width=0.5,color=['blue','orange'],linewidth=4)
  st.write(fig)
st.write('In the other hand we have too much predictors for the amount of data that we are using and I am afraid that we might me overfitting the model with the chose of the margin of decision 0.9')

st.subheader('Resampling to balance the data')

st.write('We will use resampling techniques to balance the data. Specifically we will use **Upsampling** and **SMOTENC**. Once done the resampling we get better balanced data:')
dades_train_UCI_No=dades_train[dades_train['UCI']=='No']
dades_train_UCI_Yes=dades_train[dades_train['UCI']=='Yes']
dades_UCI_Yes_upsampled=resample(dades_train_UCI_Yes,replace=True,n_samples=len(dades_train_UCI_No),random_state=123)
dades_train_upsampled=pd.concat([dades_train_UCI_No,dades_UCI_Yes_upsampled])

categ_indexes=[0,2,7,8]+list(range(10,dades_train.shape[1]-1))
smot = SMOTENC(categorical_features=categ_indexes, random_state=123)
dades_train_SMOTENC, dades_train_SMOTENC_UCI = smot.fit_resample(dades_train.drop('UCI',axis=1), dades_train['UCI'])
dades_train_SMOTENC['UCI']=dades_train_SMOTENC_UCI

col_7, col_8=st.columns(2)
with col_7:
  fig, ax=plt.subplots(figsize=(8,5))
  plt.bar(['No','Si'],[len(dades_train_upsampled[dades_train_upsampled['UCI']=='No']),len(dades_train_upsampled[dades_train_upsampled['UCI']=='Yes'])],width=0.5,color=['blue','green'],linewidth=4)
  plt.title('Upsampled data')
  st.write(fig)
with col_8:
  fig, ax=plt.subplots(figsize=(8,5))
  plt.bar(['No','Si'],[len(dades_train_SMOTENC[dades_train_SMOTENC['UCI']=='No']),len(dades_train_SMOTENC[dades_train_SMOTENC['UCI']=='Yes'])],width=0.5,color=['blue','orange'],linewidth=4)
  plt.title('SMOTENC data:')
  st.write(fig)

st.write('Now we will use this new data to train the models')
st.subheader('Training logistic regresion with the Upsampled data:')
st.write('First we train the model with all the variables and we will eliminate the ones with a p-value higher than 0.05')

model_log_upsampling=smf.glm(formula=formula,data=dades_train_upsampled,family=sm.families.Binomial()).fit()
with st.expander('Summary of the model'):
  st.write(model_log_upsampling.summary())
st.write('Then we can eliminate all the variables that are not helping us and train another model:')
formula_simple='UCI~BW_2500+Gender+Hipertension+Obesity+Age+IMC'
model_log_upsampling_simple=smf.glm(formula=formula_simple,data=dades_train_upsampled,family=sm.families.Binomial()).fit()
with st.expander('Summary of the simplyfied model'):
  st.write(model_log_upsampling_simple.summary())
st.write('And we obtain our predictions:')

with st.expander('Check the different precisions'):
  prediccions_prob=model_log_upsampling_simple.predict(dades_test)
  nombres=[]
  for i in range(1,20):
    nombres.append(i/20)
  for i in nombres:
    prediccions=['No' if x>i else 'Yes' for x in prediccions_prob]
    st.write('Margin of decission '+'**'+str(i)+'**')
    pr_yes, pr_no, pr_tot=af.precisions(dades_test['UCI'],prediccions)
    st.write(pr_yes)
    st.write(pr_no)
    st.write(pr_tot)
    st.write('________________________________________________')
st.write('The best results that we get considering that our interest is in predicting tha cases that the patient will end up in the UCI are with a margin of 0.5')

st.subheader('Training logistic regression with the SMOTENC data:')
st.write('First we train our model with all the variables and we eliminate the ones with a p-values higher than 0.05')

variables='+'.join(dades_train_SMOTENC.columns.difference(['UCI','BW','IUGR_missing']))
formula='UCI~'+variables
model_log_SMOTENC=smf.glm(formula=formula,data=dades_train_SMOTENC,family=sm.families.Binomial()).fit()

with st.expander('Summary of the model'):
  st.write(model_log_SMOTENC.summary())
st.write('We eliminate the predictors that are not aporting information and we train a new model:')
formula_simplificada_SMOTENC='UCI~BW_2500+Cancer+DM+Dyslipidemia+Gender+Hipertension+Obesity+Psychiatric+Age'
model_log_SMOTENC_simplificat=smf.glm(formula=formula_simplificada_SMOTENC,data=dades_train_SMOTENC,family=sm.families.Binomial()).fit()

with st.expander('Summary of the simplyfied model'):
  st.write(model_log_SMOTENC_simplificat.summary())
st.write('And we obtain the predictions')

with st.expander('Check the different precisions'):
  prediccions_prob=model_log_SMOTENC_simplificat.predict(dades_test)
  nombres=[]
  for i in range(1,20):
    nombres.append(i/20)
  for i in nombres:
    prediccions=['No' if x>i else 'Yes' for x in prediccions_prob]
    st.write('Margin of decission '+'**'+str(i)+'**')
    pr_yes, pr_no, pr_tot=af.precisions(dades_test['UCI'],prediccions)
    st.write(pr_yes)
    st.write(pr_no)
    st.write(pr_tot)
    st.write('________________________________________________')
st.write('It looks that this model is not working as good as we expected, but the best results we get are with the margin of 0.7')

st.subheader('Random forest with the library h2o')
st.write('We will be using the library h2o for the RF model because this library works very well with categorical values. We will train two different models with the two different resampled data and we will be using the same important variables that we got in the logistic regressions. We will use the next hiperparameters:')
st.write('* **ntrees=500**, we will train a high amount of trees to avoid overfitting')
st.write('* **max_depth=20**, we will set the maximum depth of the trees to be 20 because of the numer of predictors that we have')
st.write('* **min_rows=20**, minimum of patients that has to contain a node to split')
st.write('* **mtries=-1**, the amount of random predictors used to grow the trees is set to p^(1/2) beeing p the number of predictors. This is generaly the best choice for classification problems')

st.write('If we run the code we get the following results:')
with st.expander('Results:'):
  st.write('Upsampling')
  st.write('Precision for the Yes: 0.7272727272727273')
  st.write('Precision for the No: 0.7647058823529411')
  st.write('Total precission: 0.759493670886076')
  st.write('___________________________________')
  st.write('SMOTENC:')
  st.write('Precision for the Yes: 0.7272727272727273')
  st.write('Precision for the No: 0.6323529411764706')
  st.write('Total precission: 0.6455696202531646')

st.write('**Very important:** The library h2o is not working in VS and I have tried to fix it but nothing seems to work, the code of this part will be in a colab notebook in the repository of the project named **RandomForest_Code and NN**')

st.header('Deep learning model of the data using Neural Networks (nn)')
st.write('To end the project we will try to fit a simple neural network to our data to see if we can improve the results. The fist thing we have to do is change the captegorical predictors to numerical ones and create PyTorch tensors from the data')
st.write('Once done this we will train the network with the upsampled data and SMOTENC data. For the moment we will use all the predictors. If we look at the code we will see that the used activation function for the layer is the ReLu function that is reccomended for classification problems.')

with st.expander('Graphic of the activation ReLU function'):
  x=np.linspace(-10,10,100)
  y=list(map(af.ReLU,x))
  fig,ax=plt.subplots(figsize=(8,5))
  plt.plot(x,y,'r')
  plt.title('ReLU(x)=max(0,x)')
  st.write(fig)

st.write('We will use **ADAM** optimizer with a learing rate of 0.0001 and the **negative logaritmic likehood** for the loss function. Training the model with the diferent data sets and filtering the results that fit better our objective we get the following accuracies:')
with st.expander('Best results of the NN in the validation set'):
  st.write('SMOTENC training set')
  st.write('Accuracy Yes: 0.9 with rep=29')
  st.write('Accuracy No: 0.7017543859649122 with rep=29')

  st.write('Upsampled training set')
  st.write('Accuracy Yes: 0.847457627118644 with rep=26')
  st.write('Accuracy No: 0.5416666666666666 with rep=26')

st.write('We will use this results from the validation set to adjust the hiperparameters of the NN and use the model to get predictions from the test data to get a final accuracy. We chose rep=29. We will consider just the NN trained with the SMOTENC sample because it is the one that works better')
with st.expander('Results in the Test dataset with rep=29'):
  st.write('Accuracy Yes: 0.8181818181818182 with rep=29')
  st.write('Accuracy No: 0.5147058823529411 with rep=29')

st.write('**Very important:** I have the exact same problem with the library torch in VS so the code for this part will be in a colab notebook in the repository of the project named **RandomForest_Code and NN**')

st.header('Conclusions of the project')

st.write('About the predictors looking at the summary of the upsampling logistic regression we can determine the following:')
st.write('1. Having a lower birth weight than the expected has a big impact in the probability of suffering complications')
st.write('2. Men are more likely to end up in the UCI if positive of covid')
st.write('3. Hipertension has a big impact too in the probability of suffering complications')
st.write('4. Age is obiously a factor that we have to consider')
st.write('5. The body mass index has a small impact in the probability of ending on the UCI')

st.subheader('Conclusions about the performance of the different models')
st.write('The next table displays the models that have worked better for the project')
taula=pd.DataFrame({'Yes':[0.82,0.73,0.55,0.73,0.73,0.81],'No':[0.60,0.70,0.54,0.76,0.63,0.51]})
taula.index=['Logistic reg. with waw data and predictors','Logistic reg. Upsampled','Logistic reg. SMOTENC','R.F. Upsampling','R.F. SMOTENC','Neural Network SMOTENC']
st.table(taula)
st.write('The first model has good results but we can not relly on this model because there is a high risk of overfitting. The model that seems to have the best results is Random Forest with the upsampled training set or the neural network in the case of the patients that will suffer complications. We would suggest the hospital to use a mix of the two models for predicting.')

st.write('The last conclusion is that Naural Networks have huge predicting potencial and I am sure that with a more deep study and modeling of the Net we could get better precisions for our data. If we just keep trying to train the Network without changing anything due to the randomness of the DataLoader we get precisions better that the ones we already have so that makes me conclude that with more data that might be abailable in the future we can train deep neural networks that can fit the data **up to precisions of 0.9**.')
