import numpy as np


def rejection_sampling(unnorm_pdf, xmin, xmax, k=None):
    if k is None:
        x = np.linspace(xmin, xmax, 10001, endpoint=True)
        unnorm_pdf_max = np.max(unnorm_pdf(x))
        k = unnorm_pdf_max * 1.1

    while (True):
        xtrial = np.random.uniform(xmin, xmax)
        u = np.random.uniform(0.0, k)
        if u < unnorm_pdf(xtrial):
            break

    return xtrial
