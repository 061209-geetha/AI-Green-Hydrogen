"""Attribution diagnostics; gradient saliency always available, SHAP optional."""
from pathlib import Path
import numpy as np
import pandas as pd
from .forecast import SEQ_COLS,STATIC_COLS
from .common import save_table

def gradient_saliency(model,data,masks,out,max_test=256):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idx=np.flatnonzero(masks['test'])[:max_test]
    if not len(idx):return None
    model.model.eval()
    x=torch.as_tensor(data['x'][idx],device=model.device).clone().detach().requires_grad_(True)
    s=torch.as_tensor(data['s'][idx],device=model.device).clone().detach().requires_grad_(True)
    y=model.model(x,s).sum();y.backward()
    grad=x.grad.detach().cpu().numpy()
    seq=np.mean(np.abs(grad),axis=(0,1))
    stat=np.mean(np.abs(s.grad.detach().cpu().numpy()),axis=0)
    names=[f'time:{name}' for name in SEQ_COLS]+[f'static:{name}' for name in STATIC_COLS]
    names +=[f'ACORN dummy {i}' for i in range(len(stat)-len(STATIC_COLS))]
    imp=np.concatenate([seq,stat])
    df=pd.DataFrame({'feature':names,'mean_absolute_input_gradient':imp}).sort_values('mean_absolute_input_gradient',ascending=False)
    save_table(df,Path(out)/'tables'/'gradient_saliency')
    plot=df.head(14).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6));ax.barh(plot.feature,plot.mean_absolute_input_gradient)
    ax.set_xlabel('Mean absolute input gradient (standardised features)');fig.tight_layout()
    fig.savefig(Path(out)/'figures'/'gradient_feature_importance.png',dpi=600);plt.close(fig)
    return df

def shap_attribution(model,data,masks,out,background=40,examples=60):
    """Optional PyTorch SHAP GradientExplainer; may vary by SHAP version."""
    import torch
    import shap
    class Wrapper(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,x,s):return self.net(x,s).unsqueeze(1)
    torch_model=Wrapper(model.model).to(model.device).eval()
    a=np.flatnonzero(masks['train'])[:background]
    b=np.flatnonzero(masks['test'])[:examples]
    if len(a)==0 or len(b)==0:raise ValueError('SHAP requires nonempty train and test data')
    def tens(indices):return [torch.as_tensor(data['x'][indices],device=model.device),torch.as_tensor(data['s'][indices],device=model.device)]
    values=shap.GradientExplainer(torch_model,tens(a)).shap_values(tens(b))
    # SHAP multiinput arrays: [N,lookback,features,1] and [N,static,1].
    seq,stat=values
    seq=np.asarray(seq).squeeze(-1);stat=np.asarray(stat).squeeze(-1)
    imp=np.concatenate([np.mean(abs(seq),axis=(0,1)),np.mean(abs(stat),axis=0)])
    names=[f'time:{n}' for n in SEQ_COLS]+[f'static:{n}' for n in STATIC_COLS]
    names +=[f'ACORN dummy {i}' for i in range(len(stat[0])-len(STATIC_COLS))]
    df=pd.DataFrame({'feature':names,'mean_absolute_shap':imp}).sort_values('mean_absolute_shap',ascending=False)
    save_table(df,Path(out)/'tables'/'shap_importance')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot=df.head(14).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6));ax.barh(plot.feature,plot.mean_absolute_shap)
    ax.set_xlabel('Mean absolute SHAP attribution (scaled forecast)');fig.tight_layout();fig.savefig(Path(out)/'figures'/'shap_importance.png',dpi=600);plt.close(fig)
    return df
