from transforms.api import transform, Input, Output
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# Supposons que vous ayez un dataset en entrée (même s'il n'est pas utilisé pour le graphique)
@transform(
    my_output_dataset=Output("/path/to/your/output_plot_dataset"),
    my_input_dataset=Input("/path/to/an/input_dataset")
)
def compute(my_input_dataset, my_output_dataset):
    # 1. Création du graphique
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4]
    y = [10, 20, 25, 30]
    ax.plot(x, y)
    ax.set_title("Mon Graphique Matplotlib")
    ax.set_xlabel("Axe X")
    ax.set_ylabel("Axe Y")

    # 2. Sauvegarde du graphique dans un buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # 3. Préparation pour l'écriture dans un dataset
    # Vous pouvez ajouter des métadonnées si nécessaire
    plot_df = pd.DataFrame({
        "plot_name": ["mon_graphique_simple"],
        "plot_image": [buf.getvalue()]  # Utiliser getvalue() pour obtenir les bytes
    })

    # 4. Écriture dans le dataset de sortie
    my_output_dataset.write_dataframe(plot_df)

