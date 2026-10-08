"""Paired change in ablation effects across saved training steps."""
import json
from pathlib import Path
import torch

ROOT = Path('/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/local-match-length-20261008/trajectory')
PAIRS = [('late_true', 'late_true_4000', 'late_true_6000'),
         ('shared', 'shared_6000', 'shared_12000')]
report = {}
for family, early, late in PAIRS:
    if not all((ROOT / name / 'results.json').exists() for name in (early, late)):
        continue
    results = [json.loads((ROOT / name / 'results.json').read_text()) for name in (early, late)]
    protocols = [json.loads((ROOT / name / 'protocol.json').read_text()) for name in (early, late)]
    assert results[0]['input_sha256'] == results[1]['input_sha256']
    assert protocols[0]['model_spec'] == protocols[1]['model_spec']
    assert protocols[0]['training']['run'] == protocols[1]['training']['run']
    batches = [json.loads((ROOT / name / 'batches.json').read_text()) for name in (early, late)]
    assert [r['tokens'] for r in batches[0]] == [r['tokens'] for r in batches[1]]
    count = torch.tensor([r['tokens'] for r in batches[0]], dtype=torch.float64)
    index = torch.randint(len(count), (4000, len(count)), generator=torch.Generator().manual_seed(99))
    changes = {}
    for mode in results[0]['results']:
        effects = [torch.tensor([r['nats'][mode] - r['nats']['full'] for r in rows], dtype=torch.float64)
                   for rows in batches]
        change = effects[1] - effects[0]
        bootstrap = change[index].sum(1) / count[index].sum(1)
        changes[mode] = dict(early=effects[0].sum().item()/count.sum().item(),
                             late=effects[1].sum().item()/count.sum().item(),
                             change=change.sum().item()/count.sum().item(),
                             change_ci95=torch.quantile(bootstrap, torch.tensor([.025, .975], dtype=torch.float64)).tolist())
    report[family] = dict(steps=[p['step'] for p in protocols],
                          packed_training_tokens=[p['step']*p['training']['total_batch_size'] for p in protocols],
                          nll=[r['results']['full']['nll'] for r in results],
                          valid_tokens=results[0]['valid_tokens'], input_sha256=results[0]['input_sha256'],
                          effects=changes,
                          caveat='Two checkpoints, one seed per trajectory; training progress includes hard annealing and LR changes. CI is paired over packed batches.')
    report[family]['position_effects'] = [
        dict(start=a['start'], end=a['end'], tokens=a['tokens'],
             effects={mode: [row['nll'][mode]-row['nll']['full'] for row in (a,b)]
                      for mode in changes})
        for a,b in zip(results[0]['position'],results[1]['position'])]
    report[family]['checkpoint_smoothed_training_loss'] = [
        json.loads(Path(p['metadata']).read_text())['loop_state']['smooth_train_loss']
        for p in protocols]
(ROOT / 'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
