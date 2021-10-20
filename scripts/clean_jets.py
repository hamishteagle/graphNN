from collections import OrderedDict
import numpy as np
import pprint as pp
import NN_driver as driver  # The variable names are stored in the driver config

"""Script to simplifiy the jet variables in the inputs"""


# Load up the uncleaned datasets (still contain additional jet variables)
x_train = np.load("data/X_train.npy")
x_test = np.load("data/X_test.npy")


variables = driver.variables


def clean(x, variables):
    assert len(x) == len(variables)

    var_map = OrderedDict()
    for var, name in zip(x, variables):
        var_map[name] = var

    # Figure out which jet is non b-tagged, remove the extra variables
    jet_options = ["j1_quantile", "j2_quantile", "j3_quantile"]
    for j_q in jet_options:
        if var_map[j_q] >= 3:
            bjet = j_q.replace("_quantile", "")  # extract which number jet is a b-jet, remove these as they are already filled by b1* b2* variables
            del var_map[j_q]
            del var_map["phi" + bjet]
            del var_map["eta" + bjet]
            del var_map["pT" + bjet]
    x = np.asarray([v for _, v in var_map.items()])
    return x


x_train = np.asarray([clean(x, variables) for x in x_train])
x_test = np.asarray([clean(x, variables) for x in x_test])

np.save("data/X_train_cleaned.npy", x_train)
np.save("data/X_test_cleaned.npy", x_test)
