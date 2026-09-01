import os
import subprocess
import tempfile

import polars as pl
from dotenv import load_dotenv

load_dotenv()


def run_backtest_parallel(
    data: pl.DataFrame,
    signal_name: str,
    constraints: list[str],
    gamma: float,
    n_cpus: int,
):
    # Get unique years from the alphas data
    years = sorted(data.select(pl.col("date").dt.year()).unique().to_series().to_list())
    num_years = len(years)

    # Get super computer job variables
    byu_email = os.getenv("BYU_EMAIL")
    project_root = os.getenv("PROJECT_ROOT")
    years_str = " ".join(str(y) for y in years)
    constraints_str = " ".join(constraints)
    temp_dir = f"{project_root}/temp"
    data_path = f"{temp_dir}/alphas.parquet"
    output_dir = f"{project_root}/weights/{signal_name}/{gamma}"
    logs_dir = f"logs/{signal_name}/{gamma}"

    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Save alphas to temporary directory
    data.write_parquet(data_path)

    # Format sbatch_script
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=bma_reversal_backtest
#SBATCH --output=logs/{signal_name}/{gamma}/backtest_%A_%a.out
#SBATCH --error=logs/{signal_name}/{gamma}/backtest_%A_%a.err
#SBATCH --array=0-{num_years - 1}%31
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-user={byu_email}
#SBATCH --mail-type=BEGIN,END,FAIL

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

DATA_PATH="{data_path}"
OUTPUT_DIR="{output_dir}"
GAMMA="{gamma}"
N_CPUS="{n_cpus}"
CONSTRAINTS="{constraints_str}"

# Years to process
years=({years_str})

num_years=${{#years[@]}}

if [ $SLURM_ARRAY_TASK_ID -ge $num_years ]; then
echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((num_years-1)))."
exit 1
fi

year=${{years[$SLURM_ARRAY_TASK_ID]}}

source {project_root}/.venv/bin/activate
echo "Running year=$year"
srun python research/utils/mvo.py --data_path "$DATA_PATH" --gamma "$GAMMA" --year "$year" --output_dir "$OUTPUT_DIR" --n_cpus "$N_CPUS" --constraints $CONSTRAINTS
    """

    # Write the script to a temporary file and submit it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(sbatch_script)
        script_path = f.name

    try:
        # Submit the job using sbatch
        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )
        print(f"Job submitted successfully!")
        print(f"sbatch output: {result.stdout}")
        if result.stderr:
            print(f"sbatch stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print(
            "Error: sbatch command not found. Are you running this on a system with SLURM?"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.unlink(script_path)
