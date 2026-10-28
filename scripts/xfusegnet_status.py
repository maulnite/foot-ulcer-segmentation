#!/usr/bin/env python3
"""Read-only x-FUSegNet workflow and per-fold training progress."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    config=json.loads((ROOT/'configs/xfusegnet.json').read_text())
    workflow=ROOT/config['workflow_dir']/'workflow.json'
    result=json.loads(workflow.read_text()) if workflow.exists() else {'status':'not_launched'}
    result={k:v for k,v in result.items() if k not in {'command','publication_base'}}
    result['folds']=[]
    run=ROOT/config['run_dir']
    for index in range(1,6):
        fold=run/f'fold-{index}'
        row={'fold':index,'status':'queued','completed_epochs':0}
        if (fold/'status.json').exists():row.update(json.loads((fold/'status.json').read_text()))
        if (fold/'history.json').exists():
            history=json.loads((fold/'history.json').read_text())
            if history:row['median_epoch_seconds']=statistics.median(r['seconds'] for r in history)
        result['folds'].append(row)
    return result


if __name__=='__main__':print(json.dumps(snapshot(),indent=2))
