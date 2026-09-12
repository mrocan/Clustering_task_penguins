import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

from FuncionesMineria2 import (plot_varianza_explicada, plot_cos2_heatmap, plot_corr_cos, plot_cos2_bars,
                               plot_contribuciones_proporcionales, plot_pca_scatter, plot_pca_scatter_with_vectors,
                               plot_pca_scatter_with_categories)

penguins = sns.load_dataset('penguins')

print(penguins.head())

penguins.describe()

# Genera una lista con los nombres de las variables.
variables = list(penguins.columns)  
 
categoricas = ['species', 'island', 'sex'] 
new_penguins = penguins.drop(categoricas, axis=1)
variables_new = list(new_penguins.columns)


# Cálculo de los estadísticos descriptivos.
estadisticos = pd.DataFrame({
    'Mínimo': new_penguins[variables_new].min(),
    'Percentil 25': new_penguins[variables_new].quantile(0.25),
    'Mediana': new_penguins[variables_new].median(),
    'Percentil 75': new_penguins[variables_new].quantile(0.75),
    'Media': new_penguins[variables_new].mean(),
    'Máximo': new_penguins[variables_new].max(),
    'Desviación Estándar': new_penguins[variables_new].std(),
    'Varianza': new_penguins[variables_new].var(),
    'Coeficiente de Variación': (new_penguins[variables_new].std() / new_penguins[variables_new].mean()),
    'Datos Perdidos': new_penguins[variables_new].isna().sum()  # Cuenta los valores NaN por variable.
})


sns.pairplot(new_penguins)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(penguins["flipper_length_mm"], penguins["bill_length_mm"], alpha=0.7, c='b')

# Etiquetas
plt.xlabel("Longitud de la aleta (mm)")
plt.ylabel("Longitud del Pico (mm)")
plt.title("Relación entre la Longitud de la aleta y del pico")

plt.show()
#al observar el dataframe new_penguins vemos que las filas 3 y 339 tienen todos los datos missing, por tanto eliminamos estas observaciones
new_penguins = new_penguins.drop([3,339])
penguins = penguins.drop([3,339])

R = new_penguins.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(R, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()

#---------------------------
penguins_estand = pd.DataFrame(
    StandardScaler().fit_transform(new_penguins),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in variables_new],  # Nombres de columnas estandarizadas
    index=new_penguins.index  # Índices (etiquetas de filas) del DataFrame
)

# Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=4) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 4 para retener el maximo de las componentes principales (maximo= numero de variables numericasº).
pca = PCA(n_components=4)

# Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(notas_estandarizadas) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(penguins_estand)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_*100

# Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

# Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)]) 

# Imprimir la tabla
print(tabla)

resultados_pca = pd.DataFrame(fit.transform(penguins_estand), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=penguins_estand.index)
# Representacion de la variabilidad explicada:   

plot_varianza_explicada(var_explicada, fit.n_components_)

########################################################################################
# Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca_2 = PCA(n_components=2)
fit_2 = pca_2.fit(penguins_estand)

# Obtener los autovalores asociados a cada componente principal.
autovalores_2 = fit_2.explained_variance_

# Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca_2.components_.T, 
                            columns = ['Autovector {}'.format(i) for i in range(1, fit_2.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in variables_new])

# Calculamos las dos primeras componentes principales
resultados_pca_2 = pd.DataFrame(fit_2.transform(penguins_estand), 
                              columns=['Componente {}'.format(i) for i in range(1, fit_2.n_components_+1)],
                              index=penguins_estand.index)

# Añadimos las componentes principales a la base de datos estandarizada.
datos_z_cp = pd.concat([penguins_estand, resultados_pca_2], axis=1)


# Cálculo de las correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = datos_z_cp.columns

# Calculamos las correlaciones y seleccionamos las que nos interesan (variables contra componentes).
correlacion = pd.DataFrame(np.corrcoef(penguins_estand.T, resultados_pca_2.T), 
                           index = variables_cp, columns = variables_cp)

n_variables = fit_2.n_features_in_
correlaciones_datos_con_cp = correlacion.iloc[:fit_2.n_features_in_, fit_2.n_features_in_:]

#####################################################################################################

cos2 = correlaciones_datos_con_cp **2
plot_cos2_heatmap(cos2)
#######################################################################################################

plot_corr_cos(fit_2.n_components, correlaciones_datos_con_cp)

##################################################################################################

plot_cos2_bars(cos2)

contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2,autovalores_2,fit_2.n_components)
######################################################################################################

plot_pca_scatter(pca_2, penguins_estand, fit_2.n_components)


##################################################
comp_ppales = pca_2.transform(penguins_estand)
penguins_estand_categ=pd.concat([penguins_estand, penguins[categoricas]], axis = 1)
plot_pca_scatter_with_categories(penguins_estand_categ, comp_ppales, fit_2.n_components, 'species')
#-------------------------------------------CLUSTERING--------------------------

#PRIMERO QUEREMOS UN METODO JERARQUICO

from scipy.spatial import distance

# Calculate the pairwise Euclidean distances
distance_matrix = distance.cdist(new_penguins, new_penguins, 'euclidean')

# The distance_matrix is a 2D array containing the Euclidean distances
# between all pairs of observations.
print("Distance Matrix:")
distance_small = distance_matrix[:5, :5]
#Index are added to the distance matrix
distance_small = pd.DataFrame(distance_small, index=new_penguins.index[:5], columns=new_penguins.index[:5])

distance_small_rounded = distance_small.round(2)
print('Distance Matrix:', distance_small_rounded)



plt.figure(figsize=(8, 6))
df_distance = pd.DataFrame(distance_matrix, index = new_penguins.index, columns = new_penguins.index)
sns.heatmap(df_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()


"""Standarizing the variables"""

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the DataFrame to standardize the columns
df_std = pd.DataFrame(scaler.fit_transform(new_penguins), columns=new_penguins.columns)

print(df_std)

# Calculate the pairwise Euclidean distances
distance_std = distance.cdist(df_std, df_std,"euclidean")

print(distance_std[:5,:5].round(2))

"""Recalculamos la matriz de distancias y la representamos con los datos estandarizados."""

plt.figure(figsize=(8, 6))
df_std_distance = pd.DataFrame(distance_std, index = df_std.index, columns = new_penguins.index)
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Calculate the linkage matrix
linkage_matrix = sch.linkage(df_std_distance, method='ward')  # You can choose a different linkage method if needed

# Create the dendrogram
dendrogram = sch.dendrogram(linkage_matrix, labels=new_penguins.index, leaf_font_size=9, leaf_rotation=90)

# Display the dendrogram
plt.show()

"""# Asignamos cada observación a uno de los 4 clústeres (nos quedamos con ese número)"""
#segun el dendograma 4 parece un numero razonable para los clusters

# Assign data points to 4 clusters

num_clusters = 3
cluster_assignments = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Display the cluster assignments
print("Cluster Assignments:", cluster_assignments)

# Display the dendrogram
plt.show()

"""# Añadimos la nueva variable a nustro data frame"""

# Create a new column 'Cluster' and assign the 'cluster_assignments' values to it
new_penguins['Cluster4'] = cluster_assignments

# Now 'df' contains a new column 'Cluster' with the cluster assignments

print(new_penguins["Cluster4"])

"""# Representación de los datos y su pertenencia a los clusters"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming 'df' is your original DataFrame with data
# 'cluster_assignments' contains cluster assignments

# Step 1: Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(new_penguins)

# Create a new DataFrame for the 2D principal components
df_pca = pd.DataFrame(data=pca_2.fit_transform(new_penguins), columns=['PC1', 'PC2'])

# Step 2: Create a scatter plot with colors for clusters
plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(cluster_assignments):
    plt.scatter(df_pca.loc[cluster_assignments == cluster, 'PC1'],
                df_pca.loc[cluster_assignments == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(new_penguins.index[i]), fontsize=8)

plt.title("2D PCA Plot with Cluster Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()



"""## Clustering no jerárquico
 
#Kmeans
"""

from sklearn.cluster import KMeans

# Set the number of clusters (k=4)
k = 3

# Initialize the KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the KMeans model to your standardized data
kmeans.fit(df_std)

# Get the cluster labels for your data
kmeans_cluster_labels = kmeans.labels_

print(kmeans_cluster_labels)

"""Repetimos el gráfico anterior con el k-means. ¿Será igual el gráfico?"""

# Step 2: Create a scatter plot with colors for clusters
plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(kmeans_cluster_labels):
    plt.scatter(df_pca.loc[kmeans_cluster_labels == cluster, 'PC1'],
                df_pca.loc[kmeans_cluster_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(penguins_estand.index[i]), fontsize=8)

plt.title("2D PCA Plot with K-means Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

"""No tiene mucho sentido esta solución. La posible explicación sería el tamaño de la muesstra. Vemos una limitación importante del método y porque en este caso sería más robusto y daría una solución mucho más satisfactoria y razonable.

El método de Elbow para hallar el número correcto de clústeres a crear.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Create an array to store the WCSS values for different values of K:
wcss = []

for k in range(2, 11):  # You can choose a different range of K values
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

"""Plot the WCSS values against the number of clusters (K) and look for the "elbow" point:"""

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

"""Otro método es el de las siluetas"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Create an array to store silhouette scores for different values of K

silhouette_scores = []

#Run K-means clustering for a range of K values and calculate the silhouette score for each K:

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_samples

"""Run K-means clustering with the optimal number of clusters (determined using the Silhouette Method) and obtain cluster labels for each data point:"""

# Assuming 'df_std_distance' is your standardized data and '4' is the optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_std)
labels = kmeans.labels_

"""Calculates silouhette scores for each clúster"""

silhouette_values = silhouette_samples(df_std, labels)
silhouette_values

plt.figure(figsize=(8, 6))

y_lower = 10
for i in range(4):
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title("Silhouette Plot for Clusters")
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.grid(True)
plt.show()

"""sort by labels para caracterizar los clusters"""

# Add the labels as a new column to the DataFrame

df_std['label'] = labels

df_std_sort = df_std.sort_values(by="label"); df_std_sort

cluster_centroids = df_std_sort.groupby('label').mean()
cluster_centroids.round(2)


new_penguins['label'] = labels

df_sort = new_penguins.sort_values(by="label")

cluster_centroids_orig = df_sort.groupby('label').mean()
cols= ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
cluster_centroids_orig.round(2)[cols]
##########3


# Agrupar por 'species' y calcular la media de las variables numéricas

penguins = sns.load_dataset('penguins')
penguins = penguins.drop(columns = ['island'])
penguins = penguins.dropna()
penguins_mean = penguins.groupby('species').mean(numeric_only=True)