def canvi_tab(elem):
  if(elem=='si (pasado o actual)'):
    return 'yes'
  if(elem=='nunca'):
    return 'no'
def canvi(elem):
  if(elem=='si'):
    return 'yes'
  if(elem=='no'):
    return 'no'
def canvi_UCI(elem):
  if(elem==1):
    return 'Yes'
  if(elem==0):
    return 'No'
def canvi_sex(elem):
  if(elem=='mujer'):
    return 'woman'
  if(elem=='hombre'):
    return 'man'

def precisions(reals,prediccions):
  cont_Yes=0
  cont_No=0
  for i in range(len(reals)):
    if (reals.iloc[i]==prediccions[i]) & (reals.iloc[i]=='Yes'):
      cont_Yes=cont_Yes+1
    if (reals.iloc[i]==prediccions[i]) & (reals.iloc[i]=='No'):
      cont_No=cont_No+1
  return 'Precision for the Yes: '+str(cont_Yes/len(reals[reals=='Yes'])),'Precision for the No: '+str(cont_No/len(reals[reals=='No'])),'Total precission: '+str((cont_Yes+cont_No)/len(reals))

#activation function for the NN:
def ReLU(x):
  if x<0:
    return 0
  if x>0:
    return x

#codification functions for the neural network
def codGender(elem):
  if elem=='man':
    return 1.
  else:
    return 0.
def codUCI(elem):
  if elem=='Yes':
    return 1.
  else:
    return 0.
def codBW(elem):
  if elem=='BW normal':
    return 0.
  else:
    return 1.
def cod(elem):
  if elem=='yes':
    return 1.
  else:
    return 0.
