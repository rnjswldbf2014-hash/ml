# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 rnjswldbf2014-hash
import os

def _legal(la):
    return [la] if isinstance(la[0], str) else list(la)

class _Episode:
    def __init__(self, ai): self._ai = ai
    def __enter__(self):
        _ml_begin_episode(self._ai._h); return self._ai
    def __exit__(self, *a):
        _ml_end_episode(self._ai._h); return False

class BlackBoxAI:
    def __init__(self, h, name=""):
        self._h = h
        self._name = name

    def rl(self, legal_actions, *env_args):
        lals = _legal(legal_actions)
        results = _ml_rl(self._h, lals, list(env_args))
        return results[0] if len(results) == 1 else results

    def sl(self, legal_actions, *args):
        lals = _legal(legal_actions)
        last = args[-1] if args else None
        if isinstance(last, (list, tuple)) and len(last) == len(lals) and all(isinstance(a, str) for a in last):
            answers, env_args = list(last), args[:-1]
        elif isinstance(last, str) and last in lals[0]:
            answers, env_args = [last], args[:-1]
        else:
            answers, env_args = [], args
        raw = _ml_sl(self._h, lals, list(env_args), answers)
        return raw[0] if len(raw) == 1 else raw

    def reward(self, score):      _ml_reward(self._h, float(score))
    def last_reward(self, score): _ml_last_reward(self._h, float(score))
    def episode(self):            return _Episode(self)

    def save(self):
        if not self._name:
            _ml_save(self._h)
            return
        weights = _ml_export_weights(self._h)
        if not weights:
            return
        meta = _ml_get_meta(self._h)
        file = f"{self._name}_ml_memory.pth"
        try:
            import torch
            from collections import OrderedDict
            state_dict = OrderedDict()
            for key, val in weights.items():
                if key.endswith('.shape') or key in ('input_size', 'optimizer_name'):
                    continue
                shape = weights.get(key + '.shape')
                if shape:
                    state_dict[key] = torch.tensor(val, dtype=torch.float32).reshape(shape)
                else:
                    state_dict[key] = torch.tensor(val, dtype=torch.float32)
            torch.save({
                'state_dict':    state_dict,
                'action_lists':  meta['action_lists'],
                'hidden_layers': meta['hidden_layers'],
                'optimizer_name': meta['optimizer_name'],
            }, file)
            print(f" [{self._name}] 저장 완료.")
        except ImportError:
            _ml_save(self._h)


def make(model_name, action_lists=None, hidden_layers=None, optimizer='adam', reset=False):
    if action_lists is None:
        all_lists = []
    elif isinstance(action_lists[0], str):
        all_lists = [list(action_lists)]
    else:
        all_lists = [list(al) for al in action_lists]
    if hidden_layers is None:
        hl = [128]
    elif isinstance(hidden_layers, int):
        hl = [hidden_layers]
    else:
        hl = list(hidden_layers)

    file = f"{model_name}_ml_memory.pth"

    if not reset and os.path.exists(file):
        try:
            import torch
            try:
                loaded = torch.load(file, weights_only=False)
            except TypeError:
                loaded = torch.load(file)

            if isinstance(loaded, dict) and 'state_dict' in loaded:
                al       = loaded.get('action_lists') or all_lists
                hl_saved = loaded.get('hidden_layers') or hl
                if isinstance(hl_saved, int): hl_saved = [hl_saved]
                opt_saved = loaded.get('optimizer_name', optimizer) or optimizer

                h = _ml_make(model_name, al, hl_saved, opt_saved, 0)

                state_dict = loaded['state_dict']
                # fc1.* → hidden.0.* for old files
                fixed = {}
                for k, v in state_dict.items():
                    fixed[k.replace('fc1.', 'hidden.0.') if k.startswith('fc1.') else k] = v
                state_dict = fixed

                input_size = None
                for k, v in state_dict.items():
                    if 'hidden.0.weight' in k:
                        input_size = v.shape[1]; break
                if input_size is None:
                    for k, v in state_dict.items():
                        if '.weight' in k and hasattr(v, 'shape') and len(v.shape) == 2:
                            input_size = v.shape[1]; break

                if input_size is not None:
                    flat = {k: v.detach().float().flatten().tolist()
                            for k, v in state_dict.items()}
                    _ml_import_weights(h, input_size, flat)
                    print(f" [{model_name}] 이전 학습 데이터를 성공적으로 불러왔습니다!")
                    return BlackBoxAI(h, model_name)
        except Exception:
            pass  # fall through to D's own loader

    h = _ml_make(model_name, all_lists, hl, optimizer, int(reset))
    return BlackBoxAI(h, model_name)


def resset(model_name): _ml_resset(model_name)
reset = resset

import builtins
builtins.resset = resset
builtins.reset  = resset
