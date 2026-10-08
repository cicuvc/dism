"""User-approved, per-kernel ARRIVES racecheck classification."""
import re


def classify_racecheck(status, log, sass):
    arrives = [line.strip() for line in sass.splitlines() if 'ARRIVES.LDGSTSBAR' in line]
    offsets = [re.search(r'\[UR\d+\s*\+\s*0x([0-9a-f]+)\]', line, re.I)
               for line in arrives]
    immediate = any(match and int(match[1], 16) != 0 for match in offsets)
    if status == 0 and re.search(r'RACECHECK SUMMARY: 0 hazards displayed', log):
        verdict = 'pass'
    elif (status == 99 and immediate and
          re.search(r'RACECHECK SUMMARY:.*\([1-9][0-9]* errors?', log) and
          re.search(r'\d+ passed', log) and not re.search(r'\d+ failed|Target application returned', log)):
        verdict = 'accepted_arrives_immediate_warning'
    else:
        verdict = 'fail'
    return dict(status=status, verdict=verdict, arrives=arrives,
                has_immediate_address=immediate)
