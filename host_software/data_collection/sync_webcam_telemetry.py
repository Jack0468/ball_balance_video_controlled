import argparse
import os
import sys

import numpy as np
import pandas as pd


def sync_webcam_telemetry(frame_timestamps_path, telemetry_path, output_path, max_diff_ms=None):
    df_frames = pd.read_csv(frame_timestamps_path)
    df_telemetry = pd.read_csv(telemetry_path)

    if 'frame_index' not in df_frames.columns or 'frame_timestamp_ms' not in df_frames.columns:
        raise ValueError('frame_timestamps.csv must contain frame_index and frame_timestamp_ms columns')
    if 'host_timestamp_ms' not in df_telemetry.columns:
        raise ValueError('telemetry.csv must contain host_timestamp_ms column')

    df_frames = df_frames.sort_values(by='frame_index').reset_index(drop=True)
    df_telemetry = df_telemetry.sort_values(by='host_timestamp_ms').reset_index(drop=True)

    frame_times = df_frames['frame_timestamp_ms'].to_numpy(dtype=np.int64)
    telemetry_times = df_telemetry['host_timestamp_ms'].to_numpy(dtype=np.int64)

    if len(telemetry_times) == 0:
        raise ValueError('Telemetry CSV is empty')

    indices = np.searchsorted(telemetry_times, frame_times, side='left')
    indices = np.clip(indices, 0, len(telemetry_times) - 1)

    left_closer = (
        (indices > 0)
        & (np.abs(frame_times - telemetry_times[indices - 1]) < np.abs(frame_times - telemetry_times[indices]))
    )
    indices[left_closer] -= 1

    df_synced = df_telemetry.iloc[indices].copy().reset_index(drop=True)
    df_synced.insert(0, 'frame_index', df_frames['frame_index'])
    df_synced.insert(1, 'frame_timestamp_ms', df_frames['frame_timestamp_ms'])

    if max_diff_ms is not None:
        diffs = np.abs(df_frames['frame_timestamp_ms'].to_numpy(dtype=np.int64) - df_synced['host_timestamp_ms'].to_numpy(dtype=np.int64))
        bad = diffs > max_diff_ms
        if bad.any():
            print('Warning: Some frames were aligned to telemetry rows more than', max_diff_ms, 'ms away')
            print('Max difference:', int(diffs.max()), 'ms')
            print('Frame indices with large gap:', df_frames['frame_index'][bad].tolist())

    df_synced.to_csv(output_path, index=False)
    print(f'Wrote synced telemetry to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Align webcam frame timestamps with STM32 telemetry.')
    parser.add_argument('--frame-timestamps', required=True, help='Path to frame_timestamps.csv')
    parser.add_argument('--telemetry', required=True, help='Path to telemetry.csv')
    parser.add_argument('--output', default=None, help='Path to output synced_telemetry.csv')
    parser.add_argument('--max-diff-ms', type=int, default=None,
                        help='Optional maximum acceptable timestamp gap for frame-to-telemetry alignment')
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.frame_timestamps), 'synced_telemetry.csv')

    sync_webcam_telemetry(args.frame_timestamps, args.telemetry, output_path, args.max_diff_ms)


if __name__ == '__main__':
    main()
