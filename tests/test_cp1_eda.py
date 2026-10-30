"""Boundary and selection checks independent of the observed Chronic Wound numbers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cp1_eda import area_bins, area_statistics, background_foreground_ratio, choose_examples
import cp1_eda


def test_area_bin_boundaries_are_exhaustive_and_upper_inclusive():
    fractions=[0,.0001,.005,.0051,.01,.0101,.05,.0501,.1,.1001,1]
    result=area_bins(fractions)
    assert [r['count'] for r in result]==[1,2,2,2,2,2]
    assert sum(r['count'] for r in result)==len(fractions)
    assert sum(r['percent_of_train'] for r in result)==pytest.approx(100)


@pytest.mark.parametrize('fractions',[[np.nan],[np.inf],[-.001],[1.001]])
def test_invalid_area_is_a_failure_not_a_replacement(fractions):
    with pytest.raises(ValueError):area_statistics(fractions)
    with pytest.raises(ValueError):area_bins(fractions)


def test_empty_and_positive_scopes_have_distinct_statistics():
    assert area_statistics([0,0,.1,.3])['median']==pytest.approx(.05)
    assert area_statistics([.1,.3])['median']==pytest.approx(.2)
    empty=area_statistics([])
    assert empty['n']==0 and all(v is None for k,v in empty.items() if k!='n')
    assert sum(r['count'] for r in area_bins([]))==0
    assert all(r['percent_of_train'] is None for r in area_bins([]))
    assert background_foreground_ratio(40,0) is None
    assert background_foreground_ratio(40,2)==20


def test_quantile_examples_ignore_row_order_and_use_filename_ties():
    frame=pd.DataFrame({'sample_id':['z','b','a','d','e','f','h','g'],
                        'area_fraction':[0,.1,.1,.2,.3,.4,0,.5],
                        'empty_gt':[True,False,False,False,False,False,True,False]})
    chosen=choose_examples(frame)
    assert chosen==choose_examples(frame.sample(frac=1,random_state=19))
    assert chosen[0]['sample_id']=='a'
    assert chosen[-1]['sample_id']=='h'
    assert len({r['sample_id'] for r in chosen})==len(chosen)
    positive_ids=set(frame.loc[~frame.empty_gt,'sample_id'])
    assert all(r['sample_id'] in positive_ids for r in chosen[:-1])


def test_no_positive_examples_are_invented():
    frame=pd.DataFrame({'sample_id':['b','a'],'area_fraction':[0.,0.],'empty_gt':[True,True]})
    assert [r['sample_id'] for r in choose_examples(frame)]==['a']


def test_rebuild_protects_existing_work_but_allows_generated_eda(monkeypatch,tmp_path):
    existing=tmp_path/'notebooks/01_cp1_analysis.ipynb'
    generated=tmp_path/'notebooks/04_cp1_eda.ipynb'
    existing.parent.mkdir();existing.write_text('pre-existing user edit');generated.write_text('old EDA')
    run=tmp_path/'outputs/runs/cp1-baseline-01';run.mkdir(parents=True)
    (run/'best.pth').write_bytes(b'checkpoint placeholder')
    monkeypatch.setattr(cp1_eda,'ROOT',tmp_path)
    monkeypatch.setattr(cp1_eda,'RUN',run)
    monkeypatch.setattr(cp1_eda,'OUTPUT',tmp_path/'outputs/analysis/cp1-eda-01')
    monkeypatch.setattr(cp1_eda.subprocess,'check_output',lambda *a,**k:b'notebooks/01_cp1_analysis.ipynb\0notebooks/04_cp1_eda.ipynb\0')
    cp1_eda.capture_protected_inputs()
    generated.write_text('rebuilt EDA')
    assert cp1_eda.verify_protected_inputs()['all_unchanged']
    existing.write_text('unexpected change')
    with pytest.raises(ValueError,match='01_cp1_analysis'):
        cp1_eda.verify_protected_inputs()
