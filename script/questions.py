# Format des questions :
# - Sur la forme                : Is it trangle ?
# - Sur la couleur              : Is it red ? 
# - Combien de côtés            : How many sides ?
# - Combien de couleurs         : How many colors ?

# Réponses possibles
# - yes
# - no
# - int

# L'article dit 'The model is trained end-to-end on image-question-answer triples' donc supervisé je pense
# La question sur les pointillés est plus dure car on n'a pas de label
# On peut labeliser les figures avec un code python, que ce soit les couleurs ou le nombre de côtés
# (Par contre on peut pas labeliser le nombre de côtés par couleur)

FIGURES = ['circle','kite','parallelogram','rectangle','rhombus','square','trapezoid','triangle']
COLORS = ['blue','green','red','purple','orange','black']



def questions():
    """Fait une liste des questions possibles
    
    Exemple : La question [Is it blue and red ?] devient [blue, red]   
    """
    res = []

    for i, c1 in enumerate(COLORS) :
        res += [[c1]]
        res += [['sides', c1]]
        for c2 in COLORS[i+1:] :
            res += [[c1,c2]]

    for f1 in FIGURES :
        res += [[f1]]
        for c2 in COLORS :
            res += [[f1,c2]]

    res += [['colors']]

    res += [['sides']]
    return res

q = questions()
print(q, len(q))

    

    

