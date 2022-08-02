from types import SimpleNamespace
from pathlib import Path
import numpy as np
import json
import sys
import csv

from process_clusters_kmeans import process_clusters_kmeans

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing clusters.mat and/or subParticles.mat")
    parser.add_argument("--infile", help="Name of clustering output file to read", default='clusters.mat')
    parser.add_argument("--config", "-j", help="Path to base config template to use", default=str(Path(__file__).parent.joinpath("config.json")))
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--trials", "-n", help="Number of trials to run over each parameter vector", type=int, default=1)
    parser.add_argument("--out", "-o", help="Name of output folder to write to", default='sweep')
    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = json.load(f)
    
    field_names = ["elbow_threshold", "size_threshold", "total", "avg"]
    field_names += [f'correct{i}' for i in range(args.trials)]
    data = []
    csv_path = Path(args.input).joinpath("sweep_results.csv")
    out_path = Path(args.input).joinpath(args.out)
    out_path.mkdir(exist_ok=True)

    if not csv_path.exists():
        write_header = True
    else:
        write_header = False

    with csv_path.open("a") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        if write_header:
            writer.writeheader()

        # elbow_threshold
        for i in np.linspace(0, 0.5, 11):
            # size_threshold
            for j in np.linspace(0, 1, 11):
                cur_row = { "elbow_threshold": i, "size_threshold": j }
                config["methods"]["kmeans"][args.template]["elbow_threshold"] = i
                config["methods"]["kmeans"][args.template]["size_threshold"] = j

                config_f = Path(args.config).parent.joinpath("tmp.json")

                with config_f.open('w') as f:
                    json.dump(config, f)
                
                total_correct = 0

                for k in range(args.trials):
                    out_name = str(out_path.joinpath(f"trial_{i}_{j}_{k}"))

                    particle_args = {
                        "input": args.input,
                        "infile": args.infile,
                        "template": args.template,
                        "cluster": "kmeans",
                        "alignment": "differential_evolution",
                        "config": str(config_f),
                        "output": out_name + ".mat"
                    }

                    particle_args = SimpleNamespace(**particle_args)

                    correct, n_particles = process_clusters_kmeans(particle_args, False)
                    cur_row[f'correct{k}'] = correct
                    cur_row['total'] = n_particles
                    total_correct += correct
                
                cur_row['avg'] = total_correct // args.trials

                writer.writerow(cur_row)
