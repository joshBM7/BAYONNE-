import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Modèle Random Forest pour prédire le type de véhicule
def entrainer_modele_automobile():
    np.random.seed(42)
    n = 1000
    cylindree = np.random.uniform(1.0, 5.0, n)
    poids = np.random.uniform(800, 3000, n)
    puissance = np.random.uniform(70, 500, n)

    def generer_type(c, p, pu):
        if c > 3.0 and p > 2000 and pu > 200:
            return "SUV"
        elif c < 2.0 and p < 1500:
            return "Berline"
        else:
            return "Sport"

    types = [generer_type(c, p, pu) for c, p, pu in zip(cylindree, poids, puissance)]
    X = np.column_stack((cylindree, poids, puissance))
    y = np.array(types)

    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

modele_automobile = entrainer_modele_automobile()

# Interface Random Forest
def ouvrir_interface_random_forest():
    fenetre_rf = tk.Toplevel()
    fenetre_rf.title("Prédiction Automobile - Random Forest")

    # Ajout du bouton retour
    bouton_retour = ttk.Button(fenetre_rf, text="Retour", command=fenetre_rf.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)

    tk.Label(fenetre_rf, text="Cylindrée (L)").pack()
    entree_cylindree = tk.Entry(fenetre_rf)
    entree_cylindree.pack()

    tk.Label(fenetre_rf, text="Poids (kg)").pack()
    entree_poids = tk.Entry(fenetre_rf)
    entree_poids.pack()

    tk.Label(fenetre_rf, text="Puissance (ch)").pack()
    entree_puissance = tk.Entry(fenetre_rf)
    entree_puissance.pack()

    resultat_label = tk.Label(fenetre_rf, text="", font=("Helvetica", 12))
    resultat_label.pack(pady=10)

    def predire():
        try:
            c = float(entree_cylindree.get())
            p = float(entree_poids.get())
            pu = float(entree_puissance.get())
            prediction = modele_automobile.predict([[c, p, pu]])[0]
            resultat_label.config(text=f"Type prédit : {prediction}")
        except ValueError:
            resultat_label.config(text="⚠ Entrée invalide")

    bouton_predire = tk.Button(fenetre_rf, text="Prédire", command=predire)
    bouton_predire.pack(pady=5)

# Interface Régression Linéaire
def ouvrir_interface_regression():
    fenetre_reg = tk.Toplevel()
    fenetre_reg.title("Régression Linéaire")
    
    # Ajout du bouton retour
    bouton_retour = ttk.Button(fenetre_reg, text="Retour", command=fenetre_reg.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)
    
    # Génération de données pour la régression
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 2
    
    # Création et entraînement du modèle
    model = LinearRegression()
    model.fit(X, y)
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, y, color='blue', label='Données')
    ax.plot(X, model.predict(X), color='red', label='Régression')
    ax.set_title('Régression Linéaire')
    ax.legend()
    
    # Affichage du graphique
    canvas = FigureCanvasTkAgg(fig, master=fenetre_reg)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
    
    # Affichage des coefficients
    coef_label = tk.Label(fenetre_reg, 
                         text=f"Coefficient : {model.coef_[0][0]:.2f}\nIntercept : {model.intercept_[0]:.2f}",
                         font=("Helvetica", 12))
    coef_label.pack(pady=10)

# Interface Clustering
def ouvrir_interface_clustering():
    fenetre_cluster = tk.Toplevel()
    fenetre_cluster.title("Clustering K-Means")
    
    # Ajout du bouton retour
    bouton_retour = ttk.Button(fenetre_cluster, text="Retour", command=fenetre_cluster.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)
    
    # Génération de données pour le clustering
    np.random.seed(42)
    X = np.random.randn(300, 2)
    X[100:] += 5
    X[200:] -= 5
    
    # Création et entraînement du modèle
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
              c='red', marker='x', s=200, label='Centres')
    ax.set_title('Clustering K-Means')
    ax.legend()
    
    # Affichage du graphique
    canvas = FigureCanvasTkAgg(fig, master=fenetre_cluster)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

# Interface TimeSeries
def ouvrir_interface_timeseries():
    fenetre_ts = tk.Toplevel()
    fenetre_ts.title("TimeSeries ARIMA")
    
    # Ajout du bouton retour
    bouton_retour = ttk.Button(fenetre_ts, text="Retour", command=fenetre_ts.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)
    
    # Génération de données pour la série temporelle
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    y = np.sin(0.1 * t) + np.random.normal(0, 0.1, 200)
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y)
    ax.set_title('Série Temporelle')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Valeur')
    
    # Affichage du graphique
    canvas = FigureCanvasTkAgg(fig, master=fenetre_ts)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

# Interface Validation Croisée
def ouvrir_interface_validation():
    fenetre_val = tk.Toplevel()
    fenetre_val.title("Validation Croisée")
    
    # Ajout du bouton retour
    bouton_retour = ttk.Button(fenetre_val, text="Retour", command=fenetre_val.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)
    
    # Génération de données pour la validation croisée
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    
    # Création et validation du modèle
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5)
    
    # Affichage des résultats
    resultat_text = f"Scores de validation croisée :\n{scores}\nMoyenne : {scores.mean():.3f}"
    resultat_label = tk.Label(fenetre_val, text=resultat_text, font=("Helvetica", 12))
    resultat_label.pack(pady=20)

# Interface 2 : liste des algorithmes
def ouvrir_interface_algorithmes():
    fenetre_algo = tk.Toplevel()
    fenetre_algo.title("Algorithmes de L'Intelligence Artificielle")

    cadre_exterieur = ttk.Frame(fenetre_algo, padding=20, borderwidth=2, relief="solid")
    cadre_exterieur.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Ajout du bouton retour à l'accueil
    bouton_retour = ttk.Button(cadre_exterieur, text="Retour à l'accueil", command=fenetre_algo.destroy)
    bouton_retour.pack(anchor='nw', padx=10, pady=10)

    cadre_interieur = ttk.Frame(cadre_exterieur, padding=10)
    cadre_interieur.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def afficher_details_algo(algo):
        if algo == "Random Forest":
            ouvrir_interface_random_forest()
        elif algo == "Regression Lineaire":
            ouvrir_interface_regression()
        elif algo == "Clustering":
            ouvrir_interface_clustering()
        elif algo == "TimeSeries ARIMA":
            ouvrir_interface_timeseries()
        elif algo == "Validation Croisee":
            ouvrir_interface_validation()

    boutons = [
        ("Regression Lineaire", 0, 0),
        ("Clustering", 0, 1),
        ("TimeSeries ARIMA", 0, 2),
        ("Random Forest", 1, 0),
        ("Validation Croisee", 1, 1)
    ]

    for texte, ligne, colonne in boutons:
        bouton = ttk.Button(cadre_interieur, text=texte, command=lambda t=texte: afficher_details_algo(t))
        bouton.grid(row=ligne, column=colonne, padx=10, pady=10)

# Interface principale
fenetre_principale = tk.Tk()
fenetre_principale.title("Interface Graphique Tkinter GUI")
fenetre_principale.configure(bg="white")

# Fonction de confirmation avant de fermer
def confirmer_sortie():
    if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
        fenetre_principale.destroy()

# Cadre extérieur rose
cadre_exterieur = tk.Frame(fenetre_principale, bg="white", highlightbackground="#ff1493", highlightthickness=2)
cadre_exterieur.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Titre principal centré
label_titre = tk.Label(cadre_exterieur, text="Interface Graphique Tkinter GUI", font=("Comic Sans MS", 24, "bold"), fg="#0044ff", bg="white")
label_titre.pack(pady=(30, 30))

# Cadre intérieur rose
cadre_interieur = tk.Frame(cadre_exterieur, bg="white", highlightbackground="#ff1493", highlightthickness=2)
cadre_interieur.pack(padx=40, pady=10, fill=tk.BOTH, expand=True)

# Titre du cadre intérieur
label_algo = tk.Label(cadre_interieur, text="Algorithmes de L'Intelligence Artificielle", font=("Comic Sans MS", 22, "bold"), fg="#0044ff", bg="white")
label_algo.pack(pady=(30, 40))

# Cadre pour les boutons côte à côte
cadre_boutons = tk.Frame(cadre_interieur, bg="white")
cadre_boutons.pack(pady=(10, 30))

# Style commun boutons
style = ttk.Style()
style.configure("TButton", font=("Comic Sans MS", 20, "bold"), foreground="#0044ff")

# Bouton Entrée
bouton_entree = ttk.Button(cadre_boutons, text="Entree", command=ouvrir_interface_algorithmes, style="TButton")
bouton_entree.grid(row=0, column=0, padx=40, ipadx=30, ipady=10)

# Bouton Sortie
bouton_sortie = ttk.Button(cadre_boutons, text="Sortie", command=confirmer_sortie, style="TButton")
bouton_sortie.grid(row=0, column=1, padx=40, ipadx=30, ipady=10)

# Configuration du protocole de fermeture
fenetre_principale.protocol("WM_DELETE_WINDOW", confirmer_sortie)

fenetre_principale.mainloop()