


def vote_majority(classes):
    cdic = {}
    for i in classes:
        cdic[i] = cdic.get(i, 0) + 1
    sorted_count = sorted(cdic.items(), key=lambda p: p[1], reverse=True)

    return sorted_count[0][0]



