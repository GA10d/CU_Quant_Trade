#!/usr/bin/env python3
"""
Complete Track B runner - fixed version.
"""

from track_b_pipeline import run_track_b, plot_track_b, save_track_b_artifacts, TrackBConfig

if __name__ == '__main__':
    print("Starting Track B pipeline...")
    config = TrackBConfig(num_epochs=8)
    results = run_track_b(config)
    plot_track_b(results)
    save_track_b_artifacts(results)
    print("Track B complete! Check plots and artifacts/track_b/")
