from .baseline import Baseline,BaselineConfig
from .splitbaseline import SplitBaseline,SplitBaselineConfig

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
    elif model_name.lower() == "splitbaseline":
        config = SplitBaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = SplitBaseline(config)
        print("Create Split Baseline Model")
    else:
        config = BaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = Baseline(config)
        print("Create Baseline Model")
    print(config.__dict__)
    print(model)
    return model
