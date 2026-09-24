import io,wave,numpy as np
class AudioEncoder:
    def predict(self,audio_bytes):
        with wave.open(io.BytesIO(audio_bytes),"rb") as w:
            rate=w.getframerate();ch=w.getnchannels();x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
        if ch>1:x=x.reshape(-1,ch).mean(1)
        rms=float(np.sqrt(np.mean(x*x)+1e-9));z=float(np.mean(np.abs(np.diff(np.signbit(x)))))
        f=[rms,z,len(x)/max(rate,1),float(rate),float(x.std()),float(np.max(np.abs(x)))]
        return float(np.clip(.5+rms*.8-z*.2,0,1)),float(np.clip(.55+rms,0,.95)),f
