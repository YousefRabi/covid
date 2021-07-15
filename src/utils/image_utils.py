from pathlib import Path


def get_preds_records_from_yolo_preds_folder(yolo_preds_folder):
    preds_records = []
    for path in Path(yolo_preds_folder).rglob('*.txt'):
        with open(path) as f:
            bbox_preds = f.read().splitlines()
            bbox_preds = [bbox_pred.split() for bbox_pred in bbox_preds]
            for bbox_pred in bbox_preds:
                bbox_pred = [float(x) for x in bbox_pred]
            preds_records.append([path.stem] + bbox_pred)
    
    return preds_records
