"""Measure memory pressure without counting reclaimable file cache as a leak."""
import os, time
from pathlib import Path
import psutil

def pressure_from_values(current, limit, inactive_file, reserve):
    working = max(0, current - inactive_file)
    return {'current_gib': current/2**30, 'limit_gib':limit/2**30,
            'reclaimable_gib':inactive_file/2**30, 'working_gib':working/2**30,
            'available_gib':max(0,limit-working)/2**30,
            'critical': limit-working < reserve}

def snapshot(reserve_gib=1.5):
    vm = psutil.virtual_memory()
    result = {'pid':os.getpid(), 'rss_gib':psutil.Process().memory_info().rss/2**30,
              'available_gib':vm.available/2**30, 'limit_gib':vm.total/2**30,
              'source':'system', 'critical':vm.available < reserve_gib*2**30}
    for base, used, maximum, reclaimable in [
        (Path('/sys/fs/cgroup'),'memory.current','memory.max','inactive_file'),
        (Path('/sys/fs/cgroup/memory'),'memory.usage_in_bytes','memory.limit_in_bytes','total_inactive_file')]:
        try:
            current=int((base/used).read_text());limit=int((base/maximum).read_text())
            stats=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
            if limit < 2**60:
                result.update(pressure_from_values(current,limit,int(stats.get(reclaimable,0)),reserve_gib*2**30))
                result['source']='cgroup_working_set'
                break
        except (OSError,ValueError):
            continue
    return result

_last_time=0.0
_last_result=False
_hits=0

def resource_stop(c, device=None, synchronize=False):
    """No inherited timer/fixed22GiB cap. Require persistent physical RAM pressure."""
    global _last_time,_last_result,_hits
    now=time.monotonic()
    if now-_last_time<2:return _last_result
    _last_time=now
    info=snapshot(c.get('memory_reserve_gib',1.5))
    _hits=_hits+1 if info['critical'] else 0
    _last_result=_hits>=3
    if _hits:
        from common import write_json
        path=Path(c['out'])/'memory'/f'pressure_{os.getpid()}.json'
        write_json(path,dict(info,consecutive_checks=_hits,paused=_last_result))
        print('RAM pressure: '+str(info),flush=True)
    return _last_result
