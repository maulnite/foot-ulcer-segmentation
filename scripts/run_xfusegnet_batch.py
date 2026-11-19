#!/usr/bin/env python3
"""Detached original-code five-fold training, ensemble, notebook, README, GitHub."""

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(path.read_text())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_idle(device):
    if not device.startswith('cuda:'):raise ValueError('Use explicit CUDA index')
    index=device.split(':')[1]
    pids=subprocess.check_output(['nvidia-smi',f'--id={index}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    memory,utilization=subprocess.check_output(['nvidia-smi',f'--id={index}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return not pids and int(memory)<1024 and int(utilization)<10


def publish(batch,state):
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if dirty or branch!='main':
        state.update(publication_status='deferred',publication_reason='Working tree changed or branch is not main');return
    base=read(batch/'publication-base.json');files=read(batch/'publication-manifest.json')['files'];targets=[]
    if digest(ROOT/'README.md')!=base['readme_sha256']:
        state.update(publication_status='deferred',publication_reason='README changed since workflow launch');return
    for row in files:
        relative=Path(row['destination'])
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or (relative!=Path('README.md') and relative.parts[0] not in {'reports','notebooks'})
                or relative.suffix not in {'.md','.csv','.json','.ipynb'}):raise ValueError('Invalid publication target')
        source=batch/row['source'];target=ROOT/relative
        if target.exists() and relative!=Path('README.md') and source.read_bytes()!=target.read_bytes():
            state.update(publication_status='deferred',publication_reason=f'Existing result differs: {relative}');return
        targets.append((source,target,str(relative)))
    for source,target,_ in targets:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    with (batch/'publication.log').open('a') as log:
        subprocess.run(['git','add','--',*[r[2] for r in targets]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Report completed x-FUSegNet reproduction and executed analysis'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=subprocess.run(['git','-c','credential.helper=','-c','credential.helper=!gh-orie auth git-credential','push','origin','main'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state.update(publication_status='pushed' if result.returncode==0 else 'push_failed',
                 result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=read(ROOT/'configs/xfusegnet.json');batch=ROOT/config['workflow_dir'];batch.mkdir(parents=True,exist_ok=True)
    lock=(batch/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=batch/'workflow.json'
    if path.exists() and not args.resume:raise FileExistsError('Inspect existing workflow; use --resume if appropriate')
    if not (batch/'publication-base.json').exists():
        write(batch/'publication-base.json',{'readme_text':(ROOT/'README.md').read_text(),'readme_sha256':digest(ROOT/'README.md')})
    state={'status':'starting','supervisor_pid':os.getpid(),'device':args.device,'started_at_utc':utc(),
           'publication_status':'pending','supervisor_sha256':digest(Path(__file__))}
    def save():state['updated_at_utc']=utc();write(path,state)
    def execute(command,stage,log_name):
        state.update(status=stage,command=command)
        with (batch/log_name).open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;save();next_check=time.monotonic()
            while process.poll() is None:
                if time.monotonic()>=next_check:
                    from xfusegnet_status import snapshot
                    check={'checked_at_utc':utc(),'workflow':snapshot(),'child_alive':True}
                    write(batch/'monitor-latest.json',check)
                    with (batch/'monitor-checks.jsonl').open('a') as stream:stream.write(json.dumps(check)+'\n')
                    next_check=time.monotonic()+1800
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            code=process.returncode
        state.pop('child_pid',None)
        if code:raise RuntimeError(f'{stage} failed with code {code}; see {log_name}')
    try:
        save();run=ROOT/config['run_dir']
        completed=(run/'status.json').exists() and read(run/'status.json').get('status')=='completed'
        if not completed:
            while not gpu_idle(args.device):state['status']='waiting_for_gpu';save();time.sleep(30)
            command=[sys.executable,'-u','scripts/run_xfusegnet.py','train','--device',args.device]
            if run.exists():
                if not args.resume:raise FileExistsError('Training run already exists')
                command.append('--resume')
            execute(command,'training','training.console.log')
        if not (batch/'prediction-manifest.json').exists():
            while not gpu_idle(args.device):state['status']='waiting_for_gpu';save();time.sleep(30)
            execute([sys.executable,'-u','scripts/run_xfusegnet.py','infer','--device',args.device],'inference','inference.console.log')
        execute([str(ROOT/'.venv-analysis/bin/python'),'scripts/build_xfuseg_analysis.py'],'analysis','analysis.log')
        try:publish(batch,state)
        except Exception as error:state.update(publication_status='failed',publication_error=f'{type(error).__name__}: {error}')
        state.update(status='completed',finished_at_utc=utc());save()
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at_utc=utc());save();raise


if __name__=='__main__':main()
