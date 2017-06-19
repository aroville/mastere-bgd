L'organisation du dossier est relativement simple: les fichiers sources "etape<n>.py" décrivent l'évolution itérative du code.
Chaque fichier "etape<n>.py" reprend le code de "etape<n-1>.py", et y ajoute les fonctionnalités demandées à la question correspondante.
L'étape 6 est manquante, car il m'a semblé qu'elle était étroitement liée à ce qui était demandé dans l'étape 5. Le code correspondant à ces questions est donc écrit dans "etape5.py".

Enfin, les informations que j'ai récupérées pour chaque brevet sont:
    - le titre,
    - les inventeurs,
    - la date de publication,
    - la personne en charge ("assignee")
Ces catégories d'information ont été choisies de manière arbitraire, il était possible d'en récupérer plus. Je ne l'ai pas fait car je ne pense pas que ce soit l'idée de l'exercice. On peut d'ailleurs remarquer que certains champs ne sont pas renseignés, comme par exemple le champ "Assignee:" pour certaines lignes du fichier de résultats, "patents.json".
J'ai utilisé BeautifulSoup et requests pour la récupération et le traitement du HTML, pandas pour le stockage sous format JSON.