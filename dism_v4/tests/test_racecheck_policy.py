from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.racecheck_policy import classify_racecheck


def test_racecheck_acceptance_policy():
    log = '2 passed\nRACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)'
    for suffix in ('error', 'errors'):
        case = log.replace('1 error,', f'1 {suffix},')
        direct = 'ARRIVES.LDGSTSBAR.64.TRANSCNT [UR4];'
        offset = 'ARRIVES.LDGSTSBAR.64.TRANSCNT [UR7+0x8];'
        assert classify_racecheck(99, case, direct)['verdict'] == 'fail'
        assert classify_racecheck(99, case, offset)['verdict'] == 'accepted_arrives_immediate_warning'
        assert classify_racecheck(1, case, offset)['verdict'] == 'fail'
        assert classify_racecheck(99, case + '\n1 failed', offset)['verdict'] == 'fail'
    assert classify_racecheck(0, 'RACECHECK SUMMARY: 0 hazards displayed', '')['verdict'] == 'pass'
