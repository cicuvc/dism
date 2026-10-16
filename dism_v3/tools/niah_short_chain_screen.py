"""Follow-up to collision screen: remove length <=2 edges at single queries.

short_candidate uses the annotated source protection; min_chain3 does not use
the source location and removes all short matches, including useful ones.
This is an opt-in intervention, not a change to production matching semantics.
"""
import hashlib
import json
from pathlib import Path
import sys

import niah_merge_screen as base

original = base.interventions


def interventions(lengths, tau, features, source, candidate, pool, query, seed):
    deltas, diagnostics = original(lengths, tau, features, source, candidate, pool, query, seed)
    mass, _, _ = base.normalized(lengths, tau)
    reference = mass @ features
    short = (lengths > 0) & (lengths <= 2)
    for name, remove in [('short_candidate', candidate*short), ('min_chain3', short.astype(float))]:
        new, _, _ = base.normalized(lengths, tau, 1.-remove)
        deltas[name] = new @ features-reference
    diagnostics['short_candidate_mass'] = float(mass @ (candidate*short))
    diagnostics['short_source_mass'] = float(mass @ (source*short))
    return deltas, diagnostics


if __name__ == '__main__':
    base.interventions = interventions
    base.MODES = (*base.MODES, 'short_candidate', 'min_chain3')
    base.main()
    out = Path(sys.argv[sys.argv.index('--out')+1])
    (out/'followup_manifest.json').write_text(json.dumps(dict(
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        argv=sys.argv, note=__doc__), indent=2)+'\n')
