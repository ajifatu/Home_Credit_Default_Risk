import seaborn as sns 
import matplotlib.pyplot as plt 



# AFFICHER LES DISTRIBUTIONS (HISTOGRAMMES  ET BOXPLOTS)

def distributions(df, nrows=1, ncols=2):
  """
  Affiche histogrammes et boxplotS  des colonnes numériques d'un dataframe
  """
  for col in df.select_dtypes('number').columns:
    fig, axes = plt.subplots(nrows, ncols)
    sns.histplot(df[col], bins= 20,ax = axes[0])
    sns.boxplot(x = df[col], ax = axes[1], showmeans=True)
    plt.show()
    plt.close(fig)