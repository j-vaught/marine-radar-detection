"""Reproducible source-site Ultralytics training and cross-site evaluation."""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import yaml


def coco_yolo_lines(coco: dict, image: dict) -> list[str]:
    """Convert COCO xywh coordinates to normalized YOLO coordinates."""
    categories = sorted(c['id'] for c in coco['categories'])
    class_ids = {category: index for index, category in enumerate(categories)}
    w, h = image['width'], image['height']
    lines = []
    for ann in coco['annotations']:
        if ann['image_id'] != image['id'] or ann.get('iscrowd', 0):
            continue
        x, y, bw, bh = ann['bbox']
        values = ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
        lines.append(f"{class_ids[ann['category_id']]} " + ' '.join(f'{v:.17g}' for v in values))
    return lines


def stage_split(coco: dict, image_dir: Path, root: Path) -> Path:
    """Stage native pixels with the common circular field of view applied."""
    from PIL import Image

    images = root / 'images'
    labels = root / 'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for stale in images.glob('*.png'):
        stale.unlink()
    for stale in labels.glob('*.txt'):
        stale.unlink()
    ids = set()
    for info in coco['images']:
        # IDs avoid basename collisions and make prediction-to-image mapping explicit.
        stem = str(info['id'])
        if stem in ids:
            raise ValueError(f'duplicate image id {stem}')
        ids.add(stem)
        pixels = np.asarray(Image.open(image_dir / info['file_name']).convert('RGB')).copy()
        height, width = pixels.shape[:2]
        if (width, height) != (info['width'], info['height']):
            raise ValueError(f"image dimensions do not match annotation: {info['file_name']}")
        yy, xx = np.ogrid[:height, :width]
        pixels[(xx - 531) ** 2 + (yy - 515) ** 2 > 466 ** 2] = 0
        Image.fromarray(pixels).save(images / f'{stem}.png')
        (labels / f'{stem}.txt').write_text('\n'.join(coco_yolo_lines(coco, info)))
    return images


def _model_spec(method: str, initialization: str, cfg: dict) -> str:
    variant = cfg['model']['variant']
    if initialization not in ('scratch', 'pretrained'):
        raise ValueError(f'unknown initialization: {initialization}')
    return str(Path(variant).with_suffix('.yaml' if initialization == 'scratch' else '.pt'))


def _predict(model, directory: Path, coco: dict, *, floor: float, device, imgsz: int) -> list[dict]:
    categories = sorted(c['id'] for c in coco['categories'])
    image_ids = {str(i['id']): i['id'] for i in coco['images']}
    predictions = []
    for result in model.predict(source=str(directory), conf=floor, device=device, imgsz=imgsz,
                                stream=True, verbose=False):
        if result.boxes is None:
            continue
        for xyxy, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy(), strict=True):
            x1, y1, x2, y2 = map(float, xyxy)
            predictions.append({'image_id': image_ids[Path(result.path).stem],
                                'category_id': categories[int(cls)],
                                'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(score)})
    return predictions


def run_job(job: dict) -> dict:
    """Fit once, tune on source validation, and evaluate requested fixed test sites."""
    import torch
    import ultralytics
    from ultralytics import YOLO, RTDETR
    from . import protocol, evaluation

    method = job['method'].lower()
    if method not in ('yolo', 'rtdetr'):
        raise ValueError(f'unsupported method: {method}')
    prepared = protocol.prepare_job(job)
    out = Path(job['out_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(__file__).parents[1] / method / 'config.yaml').read_text())
    for section, values in job.get('config_overrides', {}).items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    train, val = prepared['train'], prepared['validation']
    data_root = out / 'data'
    validation_gt_path = out / 'validation_gt.json'
    protocol.save_json(validation_gt_path, val)
    train_dir = stage_split(train, Path(prepared['image_dir']), data_root / 'train')
    val_dir = stage_split(val, Path(prepared['image_dir']), data_root / 'val')
    categories = sorted(train['categories'], key=lambda c: c['id'])
    if [c['id'] for c in categories] != sorted(c['id'] for c in val['categories']):
        raise ValueError('training and validation categories differ')
    data_yaml = data_root / 'data.yaml'
    data_yaml.write_text(yaml.safe_dump({'path': str(data_root), 'train': str(train_dir),
        'val': str(val_dir), 'names': {i: c['name'] for i, c in enumerate(categories)}}))
    protocol.set_seed(int(job['seed']))
    spec = _model_spec(method, job['initialization'], cfg)
    if job['initialization'] == 'pretrained':
        weights_dir = job.get('weights_dir') or job.get('weights_cache')
        if not weights_dir:
            raise ValueError('pretrained initialization requires a staged weights_dir or weights_cache')
        spec = str(Path(weights_dir) / Path(spec).name)
        if not Path(spec).is_file():
            raise FileNotFoundError(f'pre-staged checkpoint missing: {spec}')
    requested_device = str(job['device'])
    is_cuda = requested_device.startswith('cuda') or requested_device.isdigit()
    if is_cuda:
        if torch.cuda.device_count() != 1:
            raise RuntimeError('GPU worker must see exactly one GPU')
        gpu_uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not gpu_uuid.startswith('GPU-') or ',' in gpu_uuid:
            raise RuntimeError('GPU worker must be pinned with CUDA_VISIBLE_DEVICES to one GPU UUID')
        actual_uuid = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid)
        if actual_uuid.lower() != gpu_uuid.lower():
            raise RuntimeError(f'visible device UUID {actual_uuid} differs from assigned {gpu_uuid}')
        device = ''
    else:
        gpu_uuid = None
        device = requested_device
    cls = YOLO if method == 'yolo' else RTDETR
    model = cls(spec)
    initial_path = getattr(model, 'ckpt_path', None)
    if job['initialization'] == 'pretrained':
        if not initial_path or not Path(initial_path).is_file():
            raise FileNotFoundError(f'cannot hash pretrained checkpoint: {initial_path}')
        initial_hash = protocol.sha256_file(Path(initial_path))
    else:
        initial_hash = None
    t = cfg['training']
    aug = cfg.get('augmentation', {})
    args = dict(data=str(data_yaml), epochs=int(job['epochs']), patience=int(job['patience']),
                batch=int(t['batch_size']), imgsz=int(t['imgsz']), optimizer=t['optimizer'],
                lr0=float(t.get('lr0', t.get('lr', 0.001))), lrf=float(t.get('lrf', 0.01)),
                warmup_epochs=float(t['warmup_epochs']), weight_decay=float(t.get('weight_decay', 0.0005)),
                device=device, workers=int(t['workers']), amp=bool(t.get('amp', True)),
                cache=t.get('cache', False), seed=int(job['seed']), deterministic=(method != 'rtdetr'),
                project=str(out), name='fit', exist_ok=True, pretrained=job['initialization'] == 'pretrained',
                save=True, cos_lr=t.get('lr_scheduler', '').lower() in ('cosineannealinglr', 'cosine'),
                **aug)
    state = {}
    def capture_initial(trainer):
        # This callback runs after Ultralytics rebuilds and initializes its train model.
        if is_cuda:
            trainer.validator.args.device = ''
            trainer.args.device = ''
        initial_state = out / 'initial_state.pt'
        torch.save({'model': trainer.model.state_dict(), 'architecture': trainer.model.yaml}, initial_state)
        state['initial_state_sha256'] = protocol.sha256_file(initial_state)
        state['architecture'] = trainer.model.yaml
    def capture_checkpoint(trainer):
        # on_model_save fires after save_model and before final_eval strips optimizer state.
        resume_dir = out / 'resume'
        resume_dir.mkdir(exist_ok=True)
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(),
               'cuda': torch.cuda.get_rng_state_all() if is_cuda else []}
        training_state = {'epoch': trainer.epoch, 'model': trainer.model.state_dict(),
                          'optimizer': trainer.optimizer.state_dict(), 'scaler': trainer.scaler.state_dict(),
                          'scheduler': trainer.scheduler.state_dict(), 'rng': rng}
        torch.save(training_state, resume_dir / 'last_training_state.pt')
        shutil.copy2(trainer.last, resume_dir / 'last.pt')
        torch.save(rng, resume_dir / 'last_rng.pt')
        if trainer.fitness == trainer.best_fitness:
            torch.save(training_state, resume_dir / 'best_training_state.pt')
            shutil.copy2(trainer.best, resume_dir / 'best.pt')
            torch.save(rng, resume_dir / 'best_rng.pt')
            state['best_epoch'] = int(trainer.epoch)
            state['best_fitness'] = float(trainer.fitness)
    model.add_callback('on_pretrain_routine_end', capture_initial)
    model.add_callback('on_model_save', capture_checkpoint)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(**args)
    training_seconds = time.perf_counter() - started
    save_dir = Path(model.trainer.save_dir)
    best, last = save_dir / 'weights' / 'best.pt', save_dir / 'weights' / 'last.pt'
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError('training did not produce both best.pt and last.pt')
    if 'best_epoch' not in state or 'initial_state_sha256' not in state:
        raise RuntimeError('required training callbacks did not capture initialization and best checkpoint')
    best_epoch = state['best_epoch']
    selected = cls(str(best))
    post = cfg['postprocessing']
    floor = float(post['score_floor'])
    thresholds = sorted(set(map(float, post.get('score_threshold_grid', [post['score_threshold']]))))
    if not thresholds or min(thresholds) < floor:
        raise ValueError('threshold grid must be nonempty and at or above score floor')
    val_raw = _predict(selected, val_dir, val, floor=floor, device=device, imgsz=int(t['imgsz']))
    protocol.save_json(out / 'validation_raw.json', val_raw)
    sweep = []
    for threshold in thresholds:
        filtered = [p for p in val_raw if p['score'] >= threshold]
        protocol.save_json(out / f'validation_threshold_{threshold:.8g}.json', filtered)
        metrics = evaluation.evaluate_coco(val, filtered)
        sweep.append({'threshold': threshold, 'metrics': metrics})
    choice = max(sweep, key=lambda x: (x['metrics'].get('ap50', 0), x['metrics'].get('ap', 0)))
    results = {}
    for target in job.get('eval_targets', []):
        if target not in prepared['tests']:
            raise KeyError(f'unknown test target: {target}')
        coco = prepared['tests'][target]
        if sorted(c['id'] for c in coco['categories']) != [c['id'] for c in categories]:
            raise ValueError('test categories differ from training categories')
        gt_path = out / f'{target}_gt.json'
        protocol.save_json(gt_path, coco)
        target_dir = stage_split(coco, Path(prepared['image_dir']), data_root / 'test' / target)
        tick = time.perf_counter()
        raw = _predict(selected, target_dir, coco, floor=floor, device=device, imgsz=int(t['imgsz']))
        elapsed = time.perf_counter() - tick
        filtered = [p for p in raw if p['score'] >= choice['threshold']]
        protocol.save_json(out / f'{target}_raw.json', raw)
        protocol.save_json(out / f'{target}_operating.json', filtered)
        results[target] = {'ground_truth_path': str(gt_path), 'ground_truth_sha256': protocol.sha256_file(gt_path),
                           'checkpoint_sha256': protocol.sha256_file(best), 'raw_metrics': evaluation.evaluate_coco(coco, raw),
                           'operating_metrics': evaluation.evaluate_coco(coco, filtered),
                           'inference_seconds': elapsed, 'images': len(coco['images']),
                           'raw_path': str(out / f'{target}_raw.json'), 'operating_path': str(out / f'{target}_operating.json'),
                           'raw_count': len(raw), 'operating_count': len(filtered),
                           'scores': [p['score'] for p in raw]}
    resolved = vars(model.trainer.args) if hasattr(model.trainer.args, '__dict__') else dict(model.trainer.args)
    result = {'job_id': job['job_id'], 'method': method, 'source': job['source'], 'seed': job['seed'],
              'condition': job.get('condition'), 'stage': job.get('stage'), 'expected_tests': job.get('eval_targets', []),
              'manifest_sha256': protocol.sha256_file(Path(job['manifest'])),
              'annotation_sha256': protocol.sha256_file(Path(job['ann'])),
              'validation_gt_path': str(validation_gt_path), 'validation_gt_sha256': protocol.sha256_file(validation_gt_path),
              'initialization': job['initialization'], 'initial_model': spec, 'initial_sha256': initial_hash,
              'initial_state_path': str(out / 'initial_state.pt'), 'initial_state_sha256': state['initial_state_sha256'],
              'architecture': state['architecture'], 'gpu_uuid': gpu_uuid,
              'determinism_note': 'RT-DETR CUDA grid_sample backward is nondeterministic' if method == 'rtdetr' else 'deterministic algorithms requested',
              'resume_best_path': str(out / 'resume' / 'best.pt'), 'resume_last_path': str(out / 'resume' / 'last.pt'),
              'resume_best_training_state': str(out / 'resume' / 'best_training_state.pt'),
              'resume_last_training_state': str(out / 'resume' / 'last_training_state.pt'),
              'best_fitness': state['best_fitness'],
              'best_checkpoint': str(best), 'best_sha256': protocol.sha256_file(best),
              'last_checkpoint': str(last), 'last_sha256': protocol.sha256_file(last), 'best_epoch': best_epoch,
              'training_seconds': training_seconds, 'peak_vram_bytes': torch.cuda.max_memory_allocated() if is_cuda else None,
              'ultralytics_version': ultralytics.__version__, 'job': {k: str(v) if isinstance(v, Path) else v for k, v in job.items()},
              'config': cfg, 'train_args': args,
              'resolved_train_args': resolved, 'validation_sweep': sweep, 'selected_threshold': choice['threshold'],
              'validation_raw_path': str(out / 'validation_raw.json'), 'metrics_path': str(out / 'metrics.json'),
              'validation_raw_count': len(val_raw), 'validation_scores': [p['score'] for p in val_raw],
              'validation': {'raw_metrics': evaluation.evaluate_coco(val, val_raw), 'selected': choice},
              'tests': results, 'targets': results, 'training_log': str(save_dir / 'results.csv')}
    protocol.save_json(out / 'metrics.json', result)
    return result
