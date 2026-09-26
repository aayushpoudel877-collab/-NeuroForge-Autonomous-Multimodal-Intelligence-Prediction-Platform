from neuroforge.deep_learning.trainer import SyntheticMultimodalTrainer

if __name__=="__main__":
    _,metrics=SyntheticMultimodalTrainer().fit()
    print("NeuroForge neural demo metrics")
    for name,value in metrics.items(): print(f"{name}: {value:.4f}")
