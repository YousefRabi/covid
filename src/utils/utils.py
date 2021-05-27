def get_dicom_as_dict(dicom):
    res = dict()
    keys = list(dicom.keys())
    for k in keys:
        nm = dicom[k].name
        if nm == 'Pixel Data':
            continue
        val = dicom[k].value
        res[nm] = val
    return res
