from constants import Models


def clusterization_pipeline(vectors):
    r = Models.umap.fit_transform(vectors)
    c = Models.aglomerative_model.fit_predict(r)
    return r, c