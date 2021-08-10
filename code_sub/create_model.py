from transformer import Transformer,TransformerConfig
from baseline import Baseline,BaselineConfig
from splitbaseline import SplitBaseline,SplitBaselineConfig
from chi2baseline import Chi2Baseline,Chi2BaselineConfig
from fbaseline import FBaseline,FBaselineConfig
from mergesplitbaseline import MergeSplitBaselineConfig,MergeSplitBaseline
from mergeAtlas import MergeAtlasConfig, MergeAtlasBaseline

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
    elif model_name.lower() == "mergesplitbaseline":
        config = MergeSplitBaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = MergeSplitBaseline(config)
        print("Create Merge Split Baseline Model")
    elif model_name.lower() == "mergeaalhammersrbn":
        config = MergeAtlasConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = MergeAtlasBaseline(config)
        print("Create Merge Atlas Baseline Model")
    elif model_name.lower() == "chi2baseline":
        config = Chi2BaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = Chi2Baseline(config)
        print("Create chi2 Baseline Model")
    elif model_name.lower() == "fbaseline":
        config = FBaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = FBaseline(config)
        print("Create Anova F Baseline Model")
    elif model_name.lower() == "transformer":
        config = TransformerConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        config.initialize()
        model = Transformer(config)
        print("Create Transformer Model")
    else:
        config = BaselineConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = Baseline(config)
        print("Create Baseline Model")
    print(config.__dict__)
    print(model)
    return model
