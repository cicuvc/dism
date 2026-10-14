"""CPU checks for scratch-only benchmark checkpoint/Adam layout conversion."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from nanochat.models.spec import ModelSpec

spec = importlib.util.spec_from_file_location(
    'bench_conv_frontend', Path(__file__).resolve().parents[1]/'tools/bench_conv_frontend.py')
frontend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend)


def fixture():
    model_spec = ModelSpec('dism_gdn_hybrid', 1, dict(conv_impl='fused'))
    weights = {'q.conv_weight': torch.arange(12.).reshape(4,3),
               'q.weight': torch.arange(9.).reshape(3,3),
               'rng_counter': torch.tensor(76296)}
    schema = dict(optimizer='torch.optim.adamw.AdamW', groups=[
        dict(kind='adamw', parameters=['q.weight','q.conv_weight'])])
    optim = dict(param_groups=[dict(params=[7,9],weight_decay=.1,lr=.002,betas=(.9,.95))],
                 state={7:dict(step=torch.tensor(19074.),exp_avg=weights['q.weight'].clone()),
                        9:dict(step=torch.tensor(19074.),
                               exp_avg=weights['q.conv_weight']+100,
                               exp_avg_sq=weights['q.conv_weight']+200)})
    return model_spec, weights, optim, schema


def test_conversion_preserves_state_and_moments():
    args = fixture()
    before = copy.deepcopy(args)
    spec, weights, optim, schema, info = frontend.convert(*args)
    assert spec.config['conv_impl'] == 'linear_fla'
    torch.testing.assert_close(weights['q.conv.weight'][:,0,-1], args[1]['q.conv_weight'][0], rtol=0,atol=0)
    torch.testing.assert_close(weights['q.conv.weight'].squeeze(1).T.flip(0), args[1]['q.conv_weight'],rtol=0,atol=0)
    assert weights['rng_counter'].item() == 76296
    assert optim['param_groups'] == before[2]['param_groups']
    for slot in ('exp_avg','exp_avg_sq'):
        torch.testing.assert_close(optim['state'][9][slot],frontend.to_fla(args[2]['state'][9][slot]),rtol=0,atol=0)
        torch.testing.assert_close(args[2]['state'][9][slot],before[2]['state'][9][slot],rtol=0,atol=0)
    assert optim['state'][9]['step'].item() == 19074
    assert schema['groups'][0]['parameters'] == ['q.weight','q.conv.weight']
    assert args[3] == before[3]
    assert info['converted_parameters'] == ['q.conv_weight']


def test_reject_unknown_optimizer_slot():
    args = fixture()
    args[2]['state'][9]['unknown'] = torch.ones(5)
    with pytest.raises(ValueError,match='unexpected optimizer slot'):
        frontend.convert(*args)


def test_reject_already_converted():
    args = fixture()
    with pytest.raises(ValueError,match='fused source'):
        frontend.convert(ModelSpec('dism_gdn_hybrid',1,dict(conv_impl='linear_fla')),*args[1:])
