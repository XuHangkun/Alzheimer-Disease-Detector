from .baseline import Baseline,BaselineConfig

def create_model(model_name="baseline",**kwargs):
    """create model

    Args:
        model_name : name of model, default baseline
    Returns:
        model
    """
    if model_name.lower() == "baseline":
        config = BaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = Baseline(config)
        print("Create Baseline Model")
    else:
        config = BaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = Baseline(config)
        print("Create Baseline Model")
    print(config.__dict__)
    return model