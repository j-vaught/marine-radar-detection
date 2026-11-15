import json
from pathlib import Path
from unittest.mock import patch

from src.revision import fleet, jobs


def test_matrix_counts_and_targets():
    generated = jobs.standard_jobs()
    assert len(generated) == 97
    assert len({j['job_id'] for j in generated}) == 97
    assert sum(j['stage'] == 'adjacency' for j in generated) == 18
    assert sum(j['stage'] == 'diagnostic' and j['train_limit'] == 32 for j in generated) == 4
    assert all(len(j['eval_targets']) == 3 for j in generated if j['stage'] != 'diagnostic')


def test_gpu_free_rejects_busy_uuid():
    with patch.object(fleet, 'ssh', side_effect=['0, GPU-a\n1, GPU-b\n', 'GPU-a, 123\n']):
        assert fleet.gpu_free('host', {}) == [1]


def test_worker_records_model_failure(tmp_path):
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'job_id': 'bad', 'method': 'unknown', 'out_dir': str(tmp_path)}))
    dest = tmp_path / 'completion.json'
    assert fleet.worker(str(job), str(dest), 1) == 1
    completion = json.loads(dest.read_text())
    assert completion['state'] == 'failed'
    assert completion['failure_kind'] == 'model'


def test_worker_requires_metrics(tmp_path):
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'job_id': 'x', 'method': 'unet', 'out_dir': str(tmp_path)}))
    with patch('src.revision.run.run_job', return_value=None):
        assert fleet.worker(str(job), str(tmp_path / 'completion.json'), 1) == 1


def test_resolve_nested_paths():
    assert fleet.resolve({'ann': '{data}/a', 'target': ['{source}']}, {'data': '/d', 'source': 'DAAN'}) == {'ann': '/d/a', 'target': ['DAAN']}
