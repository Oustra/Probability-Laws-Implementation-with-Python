#Loi_Poisson-------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math

def PoissonCal(lamda, k):
    res = []
    for i in k:
        res.append((pow(lamda, i) * exp(-lamda)) / factorial(i))
    return res

def plot_poisson_pmf_cdf(lamda):
    # Générer des données simulées
    simulated_data = np.random.poisson(lamda, 1000)
    x = np.arange(0, 80)

    # Calculer la fonction de densité théorique
    theoretical_density = PoissonCal(lamda, x)

    plt.figure(figsize=(6, 4))
    plt.hist(simulated_data, bins= x, density=True, label=f'Simulé (λ={lamda})', alpha=0.8)
    plt.title(f'Fonction densité de la loi de Poisson (λ={lamda})')
    plt.xlabel('X')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.show()
    
    # Calculer la fonction de répartition théorique
    theoretical_cdf = np.cumsum(PoissonCal(lamda, x))
    
    plt.figure(figsize=(6, 4))
    plt.hist(simulated_data, bins= x , density=True, cumulative=True, label=f'Simulé (λ={lamda})', alpha=0.8)
    plt.title(f'Fonction de répartition de la loi de Poisson (λ={lamda})')
    plt.xlabel('X')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.show()

# Test avec différentes valeurs de λ
plot_poisson_pmf_cdf(1)
plot_poisson_pmf_cdf(15)
plot_poisson_pmf_cdf(40)






#Loi_Binomial-------------------------------------------------------------------------------------

def binomial_pmf(n, p, k):
    res = []
    for i in k:
        res.append((factorial(n) / (factorial(i) * factorial(n - i))) * (p ** i) * ((1 - p) ** (n - i)))
    return res

def binomial_cdf(n, p, k):
    pdf = binomial_pmf(n, p, k)
    cdf = np.cumsum(pdf)
    return cdf

def plot_binomial_pmf_cdf(n, p):
    # Générer des données simulées
    simulated_data = np.random.binomial(n, p, 1000)
    x = np.arange(0, n+1)

    # Calculer la fonction de densité théorique
    theoretical_density = binomial_pmf(n, p, x)

    plt.figure(figsize=(6, 4))
    plt.hist(simulated_data, bins=x - 0.5, density=True, label=f'Simulé (n={n}, p={p})', alpha=0.8)
    plt.plot(x, theoretical_density, 'r-', label='Théorique')
    plt.title(f'Fonction densité de la loi binomiale (n={n}, p={p})')
    plt.xlabel('X')
    plt.ylabel('Probabilité')
    plt.legend()
    plt.show()

    # Calculer la fonction de répartition théorique
    theoretical_cdf = binomial_cdf(n, p, x)
    
    plt.figure(figsize=(6, 4))
    plt.hist(simulated_data, bins=x - 0.5, density=True, cumulative=True, label=f'Simulé (n={n}, p={p})', alpha=0.8)
    plt.plot(x, theoretical_cdf, 'r-', label='Théorique')
    plt.title(f'Fonction de répartition de la loi binomiale (n={n}, p={p})')
    plt.xlabel('X')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.show()

# Test avec différentes valeurs de n et p
plot_binomial_pmf_cdf(50, 0.5)
plot_binomial_pmf_cdf(50, 0.85)
plot_binomial_pmf_cdf(50, 0.15)




#Loi_Normal-------------------------------------------------------------------------------------

def normal_pdf(mu, sigma, x):
    exp_term = np.exp(-0.5 * ((x - mu) / sigma)**2)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * exp_term
    return pdf

def normal_cdf(mu, sigma, x):
    standardized_x = (x - mu) / sigma
    cdf_value = 0.5 * (1 + math.erf(standardized_x / np.sqrt(2)))
    return cdf_value

def plot_normal_pdf_cdf(mu, sigma, num_samples=1000):
    # Générer des données simulées
    simulated_data = np.random.normal(mu, sigma, num_samples)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

    # Calculer la densité de probabilité (PDF) théorique
    theoretical_pdf = normal_pdf(mu, sigma, x)

    # Calculer la fonction de répartition cumulative (CDF) théorique
    theoretical_cdf = np.vectorize(lambda t: normal_cdf(mu, sigma, t))(x)

    # Tracer la densité de probabilité (PDF)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, theoretical_pdf, 'r', label='Théorique')
    plt.title('Densité de probabilité (PDF) d\'une distribution normale')
    plt.xlabel('X')
    plt.ylabel('Densité de probabilité')
    plt.legend()

    # Tracer la fonction de répartition cumulative (CDF)
    plt.subplot(1, 2, 2)
    plt.hist(simulated_data, bins=30, density=True, cumulative=True, label='Simulé', alpha=0.8)
    plt.plot(x, theoretical_cdf, 'r', label='Théorique')
    plt.title('Fonction de répartition cumulative (CDF) d\'une distribution normale')
    plt.xlabel('X')
    plt.ylabel('Probabilité cumulative')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Générer et afficher les graphiques PDF et CDF
plot_normal_pdf_cdf(20,np.sqrt(50 * 0.25), num_samples=1000)




#Loi_Exponentiel-------------------------------------------------------------------------------------

def exponential_pdf(lamda, x):
    return lamda * np.exp(-lamda * x)

def exponential_cdf(lamda, x):
    return 1 - np.exp(-lamda * x)

def plot_exponential_pdf_cdf(lamda, num_samples=1000):
    plt.figure(figsize=(12, 8))
    # Générer des données simulées
    simulated_data = np.random.exponential(scale=1/lamda, size=num_samples)
    x = np.linspace(0, 5 / lamda, 1000)

    # Calculer la fonction de densité de probabilité (PDF) théorique
    theoretical_pdf = exponential_pdf(lamda, x)

    # Calculer la fonction de répartition cumulative (CDF) théorique
    theoretical_cdf = exponential_cdf(lamda, x)

    # Tracer la fonction de densité de probabilité (PDF)
    plt.subplot(2, 2, 1)
    plt.plot(x, theoretical_pdf, label=f'Théorique (λ={lamda})')
    plt.hist(simulated_data, bins=30, density=True, alpha=0.8, label=f'Simulé (λ={lamda})')
    plt.title('Densité de probabilité (PDF) d\'une distribution exponentielle')
    plt.xlabel('X')
    plt.ylabel('Densité de probabilité')
    plt.legend()

    # Tracer la fonction de répartition cumulative (CDF)
    plt.subplot(2, 2, 2)
    plt.plot(x, theoretical_cdf, label=f'Théorique (λ={lamda})')
    plt.hist(simulated_data, bins=30, density=True, cumulative=True, alpha=0.8, label=f'Simulé (λ={lamda})')
    plt.title('Fonction de répartition cumulative (CDF) d\'une distribution exponentielle')
    plt.xlabel('X')
    plt.ylabel('Probabilité cumulative')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Générer et afficher les graphiques PDF et CDF pour les différentes valeurs de λ
plot_exponential_pdf_cdf(0.5, num_samples=1000)
plot_exponential_pdf_cdf(0.7, num_samples=1000)
plot_exponential_pdf_cdf(2, num_samples=1000)




#Temps_De_Reaction-------------------------------------------------------------------------------------

def TempsReaction(temps_reaction):
    # Calcul de la moyenne empirique
    moyenne_empirique = np.mean(temps_reaction)

    # Tracé de l'histogramme
    plt.hist(temps_reaction, bins=10, density=True, alpha=0.8)
    plt.title('Histogramme des temps de réaction')
    plt.xlabel('Temps de réaction')
    plt.show()

    # Détermination des intervalles de confiance
    Alpha1 = 0.05
    Alpha2 = 0.01

    # Utilisation de la distribution normale pour les intervalles de confiance
    n = len(temps_reaction)

    # Calcul des intervalles de confiance
    Intervalle1 = (moyenne_empirique - 1.96 * (sqrt(0.25) / sqrt(n)),
                  moyenne_empirique + 1.96 * (sqrt(0.25) / sqrt(n)))

    intervalle2 = (moyenne_empirique - 2.576 * (sqrt(0.25) / sqrt(n)),
                  moyenne_empirique + 2.576 * (sqrt(0.25) / sqrt(n)))

    print(f'Moyenne empirique : {moyenne_empirique}')
    print(f'Intervalle de confiance à 95% : {intervalle1}')
    print(f'Intervalle de confiance à 99% : {intervalle2}')

# Temps de réaction des conducteurs
temps_reaction = [0.98, 1.4, 0.84, 0.86, 0.54, 0.68, 1.35, 0.76, 0.79, 0.99,0.88, 0.75, 0.45, 1.09, 0.68, 0.60, 1.13, 1.30, 1.20, 0.91,0.74, 1.03, 0.61, 0.98, 0.91]
TempsReaction(temps_reaction)



#Estimation_une_proportion-------------------------------------------------------------------------------------

def EstimationProportion(nombre_etudiants_suivi,taille_echantillon,niveau_confiance):
    # Proportion estimée
    proportion_estimee = nombre_etudiants_suivi / taille_echantillon

    # Valeur critique z pour un niveau de confiance de 80%
    z = 1.28

    # Calcul de l'intervalle de confiance
    erreur_standard = math.sqrt((proportion_estimee * (1 - proportion_estimee)) / taille_echantillon)
    intervalle_confiance_inf = proportion_estimee - z * erreur_standard
    intervalle_confiance_sup = proportion_estimee + z * erreur_standard

    # Affichage des résultats
    print(f"Proportion estimée : {proportion_estimee}")
    print(f"Intervalle de confiance à {niveau_confiance*100}% : [{intervalle_confiance_inf}, {intervalle_confiance_sup}]")

# Données
nombre_etudiants_suivi = 673
taille_echantillon = 1000
niveau_confiance = 0.80
EstimationProportion(nombre_etudiants_suivi,taille_echantillon,niveau_confiance)


#-------------------------------------------------------------------------------------