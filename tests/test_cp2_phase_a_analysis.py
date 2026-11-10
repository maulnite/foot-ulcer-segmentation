"""Synthetic software fixtures for posthoc reporting, never experiment scores."""

import importlib.util
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name+".py")
    spec = importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    rows=[]
    for stage in (2,3):
        for group,weights in (("empty",(.2,.3,.5)),("small",(.4,.3,.3))):
            rows.append({"stage":stage,"size_group_evaluation_only":group,"fused_over_input_rms":1.5,
                         "w_additive_half":weights[0],"w_maxout":weights[1],"w_local":weights[2],
                         "normalized_gate_entropy":.9,"local_map_mean":.5,"max_gate_weight":max(weights)})
    return rows


def test_gate_statistics_keep_stage_group_units_and_explicit_absent_groups():
    result=script("analyze_cp2_phase_a").grouped_statistics(fixture_rows())
    for stage in ("2","3"):
        assert result[stage]["all"]["images"]==2
        assert result[stage]["empty"]["images"]==1
        assert result[stage]["large"]=={"images":0}
        assert result[stage]["all"]["w_local"]["mean"]==pytest.approx(.4)
        assert result[stage]["all"]["w_local"]["std"]==pytest.approx(.1)
        assert result[stage]["all"]["fraction_max_weight_over_0_95"]==0


def test_nonfusion_models_do_not_get_fabricated_gate_weights():
    rows=fixture_rows()
    for row in rows:
        for key in ("w_additive_half","w_maxout","w_local","normalized_gate_entropy","local_map_mean","max_gate_weight"):
            row[key]=None
    result=script("analyze_cp2_phase_a").grouped_statistics(rows)
    assert result["2"]["all"]["w_local"] is None
    assert result["2"]["all"]["fraction_max_weight_over_0_95"] is None


def test_case_selection_supports_executed_cohort_without_privileging_candidate():
    names=("A0","A1","A2","A3","A5","A6")
    rows={}
    for i in range(4):
        rows[str(i)]={n:{"gt_pixels":"0" if i==0 else str(i),"dice":str(i/10),
                        "tp":"0" if i==1 else "1","fp":"1" if i==0 else "0",
                        "gt_components_8_connected":"2" if i==2 else "1"} for n in names}
    choose=script("cp2_phase_a_cases").select_cases
    result=choose(rows,names)
    assert choose(dict(reversed(list(rows.items()))),names)==result
    assert result[2][1]=="1" and result[3][1]=="0" and result[4][1]=="2"
