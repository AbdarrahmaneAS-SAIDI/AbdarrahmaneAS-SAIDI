<h1 align="center">Hi 👋, I'm Abdarrahmane AS-SAIDI</h1>
<h3 align="center">👨‍💻 Final year Cybersecurity Engineering Student 🔒 Passionate about all things IT </h3> 
<!-- <p align="right"> <img src="https://komarev.com/ghpvc/?username=abdarrahmaneas-saidi&label=Profile%20views&color=0e75b6&style=flat" alt="abdarrahmaneas-saidi" /> </p> -->

Currently, I'm pursuing a FINAL-YEAR INTERNSHIP at HPS, gaining hands-on experience in securing digital infrastructures. I have a strong foundation in networks, system administration, scripting, virtualization, cloud, AI and cybersecurity best practices.

🚀 My goal: To design and implement cutting-edge security solutions as a Cybersecurity Architect. Always eager to learn and grow in the world of technology.

Let's connect! Feel free to check out my projects and collaborate on cybersecurity innovations:

<a href="https://linkedin.com/in/abdarrahmane-as-saidi"><img src="https://img.shields.io/badge/-LinkedIn-0072b1?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>

## Skills

| Skill                                         | Associated Project         |
|-----------------------------------------------|----------------------------|
| To be listed          | <a href="https://google.com">Building a Comprehensive SOC</a>|
| To be listed | <a href="https://google.com">Building a SECURE Infrastructure</a>|

## Tools
<div align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,javascript,c,java,mysql,r,go,raspberrypi,docker,aws,azure,terraform,bash,elasticsearch,grafana,kafka,kubernetes,linux,ubuntu,windows,nginx,powershell,prometheus,redhat,tensorflow" /> </br>Other Tools To be listed... </a>
</div>

## Certifications
- To be listed
<div>
<!-- <img src="https://img.shields.io/badge/-Security%2B-FF0000?&style=for-the-badge&logo=CompTIA&logoColor=white" /> -->
</div>

## Projects
- Building a Comprehensive SOC (Documentation in Progress)
- Building a SECURE Infrastructure (Documentation in Progress)



# TP1  AI

Pour accomplir cette tâche, nous devons implémenter quatre fonctions principales pour évaluer les coûts et gradients pour la régression linéaire et logistique. Ces fonctions utiliseront les conventions de forme mentionnées et des manipulations vectorielles pour éviter les boucles, en utilisant la bibliothèque NumPy. Voici les fonctions à implémenter :

1. **Fonction de coût des moindres carrés** (`mean_squared_error`)
2. **Fonction de coût de la régression logistique** (`logistic_regression`)
3. **Gradient des moindres carrés** (`mean_squared_error_gradient`)
4. **Gradient de la régression logistique** (`logistic_regression_gradient`)

### Implémentation du Code

### 1. Fonction `mean_squared_error`

Pour cette fonction, on calcule l'erreur quadratique moyenne (MSE) comme suit :
\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - X_i \cdot b)^2 \]

```python
import numpy as np

def mean_squared_error(b, x, y):
    predictions = x.dot(b)  # Prédictions linéaires
    errors = y - predictions  # Erreur entre prédictions et labels
    mse = np.mean(errors ** 2)  # Moyenne des carrés des erreurs
    return mse

```

### 2. Fonction `logistic_regression`

Pour la régression logistique, la fonction de coût est définie par :
\[ J(b) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + e^{-y_i \cdot (X_i \cdot b)}) \]

```python
def logistic_regression(b, x, y):
    logits = x.dot(b)  # Logits : produit matriciel de X et b
    cost = np.mean(np.log(1 + np.exp(-y * logits)))  # Coût logistique
    return cost

```

### 3. Gradient `mean_squared_error_gradient`

Le gradient pour la MSE est donné par :
\[ \nabla_{\text{MSE}} = -\frac{2}{n} X^T \cdot (y - X \cdot b) \]

```python
def mean_squared_error_gradient(b, x, y):
    predictions = x.dot(b)  # Prédictions
    errors = y - predictions  # Erreur
    gradient = -2 * x.T.dot(errors) / len(y)  # Gradient
    return gradient

```

### 4. Gradient `logistic_regression_gradient`

Pour la régression logistique, le gradient est :
\[ \nabla_{\text{Logistique}} = -\frac{1}{n} X^T \cdot \left(\frac{y}{1 + e^{y \cdot (X \cdot b)}}\right) \]

```python
def logistic_regression_gradient(b, x, y):
    logits = x.dot(b)
    probabilities = 1 / (1 + np.exp(y * logits))  # Sigmoïde inversée pour le gradient
    gradient = -x.T.dot(y * probabilities) / len(y)
    return gradient

```

### Vérification des Fonctions

Utilisez `check_fonctions()` pour tester les fonctions en suivant les assertions proposées.

```python
def check_fonctions():
    np.random.seed(0)
    datax = np.random.randn(1000, 2)
    datay = np.random.choice([-1, 1], size=(1000, 1))
    b_random = np.random.randn(datax.shape[1], 1)

    mse_value = mean_squared_error(b_random, datax, datay)
    assert np.isclose(mse_value, 1.59701, rtol=1e-4), f"Erreur MSE attendue: {mse_value}"

    lr_value = logistic_regression(b_random, datax, datay)
    assert np.isclose(lr_value, 0.75698, rtol=1e-4), f"Erreur logistique attendue: {lr_value}"

    mse_grad_value = mean_squared_error_gradient(b_random, datax, datay)
    assert np.isclose(mse_grad_value.mean(), 0.70793, rtol=1e-4), f"Gradient MSE moyen attendu: {mse_grad_value.mean()}"

    lr_grad_value = logistic_regression_gradient(b_random, datax, datay)
    assert np.isclose(lr_grad_value.mean(), 0.07050, rtol=1e-4), f"Gradient logistique moyen attendu: {lr_grad_value.mean()}"

    print("Toutes les fonctions ont passé les tests.")

check_fonctions()

```

Ces implémentations suivent les instructions pour éviter les boucles en utilisant des opérations matricielles et vectorielles optimisées avec NumPy, tout en respectant les formes et conventions de dimensions données pour éviter des erreurs lors de l’exécution.

Voici l'implémentation des deux fonctions demandées :

1. **`gradient_check`** : pour vérifier l'exactitude du calcul de gradient.
2. **`gradient_descent`** : pour optimiser une fonction de coût via la descente de gradient.

### 1. Implémentation de `gradient_check`

Cette fonction utilise la méthode des différences finies pour approximer le gradient numériquement, puis compare ce gradient numérique avec le gradient analytique fourni par `fn_grad`.

```python
import numpy as np

def gradient_check(fn, fn_grad, N=100, epsilon=1e-5):
    np.random.seed(0)

    # Générer des points de données et des cibles aléatoires
    datax = np.random.randn(N, 2)
    datay = np.random.randn(N, 1)
    b = np.random.randn(datax.shape[1], 1)

    # Calculer le gradient analytique
    analytic_grad = fn_grad(b, datax, datay)

    # Calculer le gradient numérique
    numeric_grad = np.zeros_like(b)

    for i in range(b.shape[0]):
        b_plus = b.copy()
        b_plus[i] += epsilon
        f_plus = fn(b_plus, datax, datay)

        b_minus = b.copy()
        b_minus[i] -= epsilon
        f_minus = fn(b_minus, datax, datay)

        numeric_grad[i] = (f_plus - f_minus) / (2 * epsilon)

    # Vérifier la similarité entre le gradient analytique et le gradient numérique
    if np.all(np.isclose(analytic_grad, numeric_grad, atol=1e-5)):
        print("Le gradient analytique et le gradient numérique sont proches pour tous les points.")
    else:
        print("Il y a une différence notable entre le gradient analytique et le gradient numérique.")

    return analytic_grad, numeric_grad

```

### 2. Implémentation de `gradient_descent`

La fonction `gradient_descent` optimise les paramètres en utilisant la descente de gradient. Elle met à jour les paramètres à chaque itération en fonction du gradient, puis stocke les valeurs de paramètres et de la fonction de coût pour chaque étape.

```python
def gradient_descent(datax, datay, fn_loss, fn_grad, eps=0.01, num_iterations=1000):
    # Initialisation aléatoire des paramètres
    b = np.random.randn(datax.shape[1], 1)
    b_values = [b.copy()]
    loss_values = [fn_loss(b, datax, datay)]

    for i in range(num_iterations):
        grad = fn_grad(b, datax, datay)
        b = b - eps * grad  # Mise à jour des paramètres avec le pas de descente
        b_values.append(b.copy())
        loss_values.append(fn_loss(b, datax, datay))  # Calcul du coût à chaque itération

        # Afficher la progression toutes les 100 itérations
        if i % 100 == 0:
            print(f"Itération {i}: Coût = {loss_values[-1]}")

    return b, b_values, loss_values

```

### Explication des Fonctions :

1. **`gradient_check`** :
    - Elle génère des points de données aléatoires et des cibles, puis compare le gradient analytique au gradient numérique obtenu par différences finies.
    - Elle utilise `np.isclose` avec une tolérance de `1e-5` pour vérifier si les gradients sont proches.
2. **`gradient_descent`** :
    - Elle initialise aléatoirement les paramètres `b`, puis réalise `num_iterations` itérations de mise à jour du gradient.
    - Elle stocke les paramètres et les valeurs de la fonction de coût à chaque étape, ce qui permet de tracer ou de vérifier la convergence après l’exécution.

### Exemple d'utilisation

Pour vérifier les fonctions `mean_squared_error` et `mean_squared_error_gradient`, nous pourrions exécuter :

```python
# Vérification du gradient pour la fonction de coût MSE
gradient_check(mean_squared_error, mean_squared_error_gradient)

# Optimisation de la fonction de coût MSE
b_optimal, b_history, loss_history = gradient_descent(datax, datay, mean_squared_error, mean_squared_error_gradient, eps=0.01, num_iterations=1000)

```

Ces fonctions devraient fournir des informations sur la progression et la convergence, et `gradient_check` permettra de confirmer la précision des gradients calculés analytiquement.
