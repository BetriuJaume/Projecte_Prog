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