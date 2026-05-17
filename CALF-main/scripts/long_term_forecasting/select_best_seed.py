#!/usr/bin/env python3
import argparse
import os
import re
import glob
import statistics


def extract_mse_from_file(path):
    # search for last occurrence of pattern like 'mse:VALUE' in file
    mse_pattern = re.compile(r'mse[:\s]*([0-9eE+\-.]+)')
    last = None
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = mse_pattern.search(line)
                if m:
                    last = float(m.group(1))
    except Exception:
        return None
    return last


def main(logs_dir, out_file, min_group_size):
    files = glob.glob(os.path.join(logs_dir, '*.logs'))
    groups = {}
    for p in files:
        b = os.path.basename(p)
        m = re.search(r'_(\d+)\.logs$', b)
        if not m:
            continue
        seed = m.group(1)
        key = re.sub(r'_(\d+)\.logs$', '', b)
        groups.setdefault(key, []).append((seed, p))

    results = []
    for key, entries in groups.items():
        if len(entries) < min_group_size:
            continue
        mses = []
        seed_to_mse = {}
        for seed, path in entries:
            mse = extract_mse_from_file(path)
            if mse is None:
                continue
            mses.append(mse)
            seed_to_mse[seed] = mse

        if not mses:
            continue

        avg_mse = statistics.mean(mses)
        # choose seed with minimum mse
        best_seed = min(seed_to_mse, key=lambda s: seed_to_mse[s])
        best_mse = seed_to_mse[best_seed]
        results.append((key, avg_mse, best_seed, best_mse, seed_to_mse))

    # write summary
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fo:
        fo.write('group,avg_mse,best_seed,best_mse,all_mses\n')
        for key, avg_mse, best_seed, best_mse, seed_map in sorted(results, key=lambda x: x[1]):
            fo.write(f"{key},{avg_mse:.6f},{best_seed},{best_mse:.6f},{seed_map}\n")

    print(f'Wrote summary for {len(results)} groups to {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', type=str, default='logs/CALF/Solar', help='directory containing .logs files')
    parser.add_argument('--out_file', type=str, default='logs/CALF/Solar/best_seeds_summary.txt', help='output summary file')
    parser.add_argument('--min_group_size', type=int, default=3, help='minimum number of seed logs to consider a group')
    args = parser.parse_args()
    main(args.logs_dir, args.out_file, args.min_group_size)
