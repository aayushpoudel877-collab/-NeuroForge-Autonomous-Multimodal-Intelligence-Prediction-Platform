import io,numpy as np
from PIL import Image
class ImageEncoder:
    def predict(self,image_bytes):
        a=np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64,64)),dtype=np.float32)/255
        mean=a.mean((0,1));std=a.std((0,1));edge=np.abs(np.diff(a,axis=0)).mean()+np.abs(np.diff(a,axis=1)).mean()
        f=np.r_[mean,std,edge,a.mean(),a.max()]
        s=float(np.clip(.5+(mean[0]-mean[2])*.5,0,1));c=float(np.clip(.55+min(std.mean(),.4),0,.95))
        return s,c,f.tolist()
