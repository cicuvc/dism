"""Read original local W&B event files, without API access or run mutation."""
import json
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

ROOT=Path(__file__).resolve().parents[2]
RUNS={'old':'run-20261022_153127-ew5yujgg','middle':'run-20261023_042852-ldu8rz57','old_planner':'run-20261024_030913-xfi42yvt'}


def main():
    result={}
    for label,name in RUNS.items():
        path=ROOT/'wandb'/name/f"run-{name.rsplit('-',1)[1]}.wandb"
        store=DataStore();store.open_for_scan(str(path));records={}
        while True:
            data=store.scan_data()
            if data is None:break
            record=wandb_internal_pb2.Record();record.ParseFromString(data)
            if not record.HasField('history'):continue
            row={}
            for item in record.history.item:
                keys=list(item.nested_key) or [item.key]
                if len(keys)!=1:continue
                key=keys[0]
                if key=='step' or key.startswith(('train/','val/','qk/')):
                    value=json.loads(item.value_json)
                    if isinstance(value,(int,float)):row[key]=value
            step=row.pop('step',None)
            if step is None:continue
            if step>6000:row={k:v for k,v in row.items() if k.startswith('val/')}
            if row:records.setdefault(int(step),{}).update(row)
        result[label]=dict(source=str(path),records=records)
        print(label,len(records),flush=True)
    out=ROOT/'dism_v3/build/early_layout_history.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
