#!/usr/bin/env python3
"""
compare_weights_header.py
Compare Keras model weights (folding BatchNorm) to arrays emitted in audio_model_weights.h
Run from repository root:
python host_software/ml_audio/models/audio_command_classifier/cpp/compare_weights_header.py

Outputs a per-layer max absolute difference and shape info.
"""
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import tensorflow as tf
except Exception as e:
    print('Missing dependency:', e)
    print('Install with: pip install numpy tensorflow')
    sys.exit(1)

# Try to locate the model and header automatically from the current working directory
cwd = Path.cwd()
found = next(cwd.rglob('best_classifier.keras'), None)
if found:
    MODEL = found
else:
    MODEL = Path(__file__).resolve().parents[2] / 'best_classifier.keras'

HEADER = Path(__file__).resolve().parent / 'audio_model_weights.h'
if not HEADER.exists():
    found_h = next(cwd.rglob('audio_model_weights.h'), None)
    if found_h:
        HEADER = found_h


def parse_header_arrays(text):
    arrs = {}
    macros = {}
    # floats
    for m in re.finditer(r'static const float ([A-Z0-9_]+)\s*\[\]\s*=\s*\{([^}]*)\};', text, re.S):
        name = m.group(1)
        vals = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', m.group(2))
        arrs[name] = np.array([float(v) for v in vals], dtype=np.float32)
    # ints
    for m in re.finditer(r'static const int ([A-Z0-9_]+)\s*=\s*([0-9]+)\s*;', text):
        macros[m.group(1)] = int(m.group(2))
    return arrs, macros


def fold_bn(kernel, bias, gamma, beta, mean, var, eps=1e-3):
    scale = gamma / np.sqrt(var + eps)
    kernel_folded = kernel * scale.reshape((1,1,1,-1))
    bias_folded = beta + (bias - mean) * scale
    return kernel_folded, bias_folded


def compare():
    if not MODEL.exists():
        print('Model not found at', MODEL)
        sys.exit(1)
    if not HEADER.exists():
        print('Header not found at', HEADER)
        sys.exit(1)

    print('Loading model:', MODEL)
    model = tf.keras.models.load_model(str(MODEL))
    text = HEADER.read_text(encoding='utf-8')
    arrs, macros = parse_header_arrays(text)
    print('Parsed header arrays:', list(arrs.keys())[:10])
    print('Parsed macros:', macros)

    layers = model.layers
    i = 0
    report = []
    conv_idx = 0
    conv1_names = ['CONV2D_KERNEL','CONV2D_1_KERNEL','CONV2D_2_KERNEL']
    while i < len(layers):
        layer = layers[i]
        cls = layer.__class__.__name__.lower()
        lname = layer.name.upper().replace('-','_').replace('/','_')
        if 'conv2d' in cls:
            kernel,bias = layer.get_weights() if layer.get_weights() else (None,None)
            next_bn = None
            if i+1 < len(layers) and 'batchnormalization' in layers[i+1].__class__.__name__.lower():
                next_bn = layers[i+1]
            if kernel is None:
                i+=1; continue
            if next_bn is not None:
                gamma,beta,mean,var = next_bn.get_weights()
                kernel,bias = fold_bn(kernel,bias,gamma,beta,mean,var,eps=getattr(next_bn,'epsilon',1e-3))
                i+=1
            kh,kw,in_ch,out_ch = kernel.shape
            # Try to find matching header array
            # Primary: NAME_KERNEL where NAME is layer.name upper
            candidates = [f"{lname}_KERNEL"] + conv1_names
            found=None
            for c in candidates:
                if c in arrs and arrs[c].size == kernel.size:
                    found=c; break
            if found is None:
                # try any conv kernel of matching size
                for kname,v in arrs.items():
                    if kname.endswith('_KERNEL') and v.size==kernel.size:
                        found=kname; break
            if found is None:
                report.append((lname,'conv', (kh,kw,in_ch,out_ch), 'MISSING_IN_HEADER', None))
            else:
                header_vals = arrs[found]
                model_flat = kernel.reshape(-1).astype(np.float32)
                # compare
                if header_vals.size < model_flat.size:
                    report.append((lname,'conv',(kh,kw,in_ch,out_ch),found,'SIZE_MISMATCH'))
                else:
                    diff = np.max(np.abs(model_flat - header_vals[:model_flat.size]))
                    report.append((lname,'conv',(kh,kw,in_ch,out_ch),found,float(diff)))
            conv_idx += 1
        elif 'dense' in cls:
            kernel,bias = layer.get_weights() if layer.get_weights() else (None,None)
            next_bn = None
            if i+1 < len(layers) and 'batchnormalization' in layers[i+1].__class__.__name__.lower():
                next_bn = layers[i+1]
            if kernel is None:
                i+=1; continue
            if next_bn is not None:
                gamma,beta,mean,var = next_bn.get_weights()
                k4 = kernel.reshape((1,1,kernel.shape[0],kernel.shape[1]))
                kf,bf = fold_bn(k4,bias,gamma,beta,mean,var)
                kernel=kf.reshape(kernel.shape)
                bias=bf
                i+=1
            in_dim,out_dim = kernel.shape
            found='DENSE_KERNEL' if 'DENSE_KERNEL' in arrs and arrs['DENSE_KERNEL'].size==kernel.size else None
            if found is None:
                # try any match
                for kname,v in arrs.items():
                    if kname.endswith('_KERNEL') and v.size==kernel.size:
                        found=kname; break
            if found is None:
                report.append((lname,'dense',(in_dim,out_dim),'MISSING_IN_HEADER',None))
            else:
                header_vals = arrs[found]
                model_flat = kernel.reshape(-1).astype(np.float32)
                diff = np.max(np.abs(model_flat - header_vals[:model_flat.size]))
                report.append((lname,'dense',(in_dim,out_dim),found,float(diff)))
        i+=1

    print('\nComparison report (layer, type, shape, header_key, max_abs_diff):')
    for r in report:
        print(r)

if __name__=='__main__':
    compare()
