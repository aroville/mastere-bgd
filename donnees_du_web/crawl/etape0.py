from tasktimer import call_repeatedly

liste = [
    "toto",
    "titi"
]


# fonction appelée périodiquement
def urlcall(to_be_processed):
    """
    Fonction appelée périodiquemnet
    :param to_be_processed: 
    :return: 
    """
    if to_be_processed["elements"]:
        call = to_be_processed["elements"].pop(0)
        print(call)
        return False
    else:
        return True

# mise en route d'un appel toutes les 5s de la fonction urlcall avec un
# dico qui contient les paramètres passés à chaque appel de la fonction
call_repeatedly(5, urlcall, {"elements": liste})



