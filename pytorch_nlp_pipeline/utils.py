import numpy as np




def get_config(params):
    # random pick a set of params
    config = {}
    
    for name in params:
        choice = np.random.choice(params[name])

        if type(choice) == np.int64:
            choice = int(choice)
        elif type(choice) == np.float64:
            choice= float(choice)
        elif type(choice) == np.str_:
            choice = str(choice)

        config[name] = choice
    

    
    return config
        