from types import SimpleNamespace
from pathlib import Path
import numpy as np
import json
import sys
import csv

from process_particles_nocluster import process_particles_nocluster
from process_particles import process_particles
from process_reads import process_reads

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to folder containing clusters.mat and/or subParticles.mat")
    parser.add_argument("--config", "-j", help="Path to base config template to use", default=str(Path(__file__).parent.joinpath("config.json")))
    parser.add_argument("--template", "-t", help="Origami template to use", default='nsf')
    parser.add_argument("--cluster", "-c", help="Whether clustering has been performed", type=bool, default=False)
    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = json.load(f)
    
    field_names = ["elbow_threshold", "size_threshold", "correct", "total"]
    data = []
    csv_path = Path(args.input).joinpath("sweep_results.csv")

    if not csv_path.exists():
        write_header = True

    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)

        if write_header:
            writer.writeheader()

        # elbow_threshold
        for i in np.linspace(0, 0.5, 11):
            # size_threshold
            for j in np.linspace(0, 1, 11):
                config["methods"]["kmeans"][args.template]["elbow_threshold"] = i
                config["methods"]["kmeans"][args.template]["size_threshold"] = j

                config_f = Path(args.config).parent.joinpath("tmp.json")

                with config_f.open('w') as f:
                    json.dump(config, f)

                out_name = f"trial_{i}_{j}"

                particle_args = {
                    "input": args.input,
                    "template": args.template,
                    "cluster": "kmeans",
                    "alignment": "differential_evolution",
                    "config": str(config_f),
                    "output": out_name + ".mat"
                }
                read_args = {
                    "input": args.input,
                    "template": args.template,
                    "infile": particle_args["output"],
                    "output": out_name
                }

                particle_args = SimpleNamespace(**particle_args)
                read_args = SimpleNamespace(**read_args)

                if args.cluster:
                    correct, n_particles = process_particles(particle_args)
                else:
                    correct, n_particles = process_particles_nocluster(particle_args)
                
                data.append({"elbow_threshold": i, "size_threshold": j, "correct": correct, "total": n_particles})
                writer.writerow(data[-1])
