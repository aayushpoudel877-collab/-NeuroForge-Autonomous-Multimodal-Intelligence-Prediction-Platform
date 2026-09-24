import base64
def validate_request(text,image_base64,audio_base64,series):
    errors=[]
    if not any([text,image_base64,audio_base64,series]): errors.append("At least one modality is required")
    if text is not None and len(text)>10000: errors.append("text exceeds 10000 characters")
    for value,label in [(image_base64,"image_base64"),(audio_base64,"audio_base64")]:
        if value:
            try: base64.b64decode(value,validate=True)
            except Exception: errors.append(f"{label} is not valid base64")
    if series is not None and any(abs(float(x))>1e12 for x in series): errors.append("series contains an out-of-range value")
    return not errors,errors
