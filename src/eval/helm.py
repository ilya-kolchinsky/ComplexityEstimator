# ---------------------------------------------------------------------------
# HELM Lite storage abstraction
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Set
import subprocess
from pathlib import Path

from src.eval.base import Example


class HelmLiteStore(ABC):
    """
    Abstract interface for accessing HELM-Lite data stored locally.

    Implementations are expected to read from local `scenario_state.json` files.

    Assumed directory layout:

        root_dir/
          <dataset_id>/
            <model_id>_scenario_state.json
            <other_model_id>_scenario_state.json
            ...

    For each dataset_id:
      - Each `<model_id>_scenario_state.json` is the raw HELM `ScenarioState` for that dataset/model.
      - All information about:
          * question text
          * full prompt sent to the model
          * model outputs
          * correctness (when derivable, e.g. MMLU multiple-choice)
        must be derived from these files.
    """

    @abstractmethod
    def load_dataset(self, dataset_id: str) -> List[Example]:
        """
        Load all examples for a given dataset_id.

        - The `query` field of each Example MUST be the full `request.prompt`
          from scenario_state.json (i.e. the **actual prompt** sent to HELM’s models).
        - The original question text SHOULD be stored in Example.metadata["question_text"].
        """
        ...

    @abstractmethod
    def get_reference_answer(self, dataset_id: str, example_id: str) -> str:
        """
        Return the reference answer for (dataset_id, example_id).

        For MMLU / multiple-choice:
        - This should be the canonical **choice index** (e.g., "A", "B", "C", "D"),
          derived from `references` and `output_mapping` inside scenario_state.json.
        """
        ...

    @abstractmethod
    def get_helm_models(self, dataset_id: str) -> Set[str]:
        """
        Return the set of model_ids that have HELM results for this dataset.

        Implementation should typically:
            - Look for `<model_id>_scenario_state.json` files under the dataset directory.
        """
        ...

    @abstractmethod
    def get_helm_answer(
        self,
        dataset_id: str,
        model_id: str,
        example_id: str,
    ) -> Optional[Tuple[str, Optional[bool]]]:
        """
        Look up the HELM answer for a given (dataset_id, model_id, example_id).

        Returns:
            (response_text, is_correct) or None if the model has no record for this example.

        - `response_text` is the raw text from the model (e.g., "A", "B", "C").
        - `is_correct` should be derived **directly from scenario_state.json** when possible
          (e.g., for MMLU: compare model's choice letter with the correct letter).
          For datasets where correctness isn't trivially derivable, implementations may
          return `is_correct=None`.
        """
        ...


# ---------------------------------------------------------------------------
# HELM / HELM Lite data download helpers
# ---------------------------------------------------------------------------

def _gcs_rsync(gcs_path: str, local_path: Path) -> None:
    """
    Internal helper: rsync from a public GCS path to a local directory
    using `gcloud storage rsync` or `gsutil rsync`.

    Requires that either `gcloud` (preferred) or `gsutil` is installed and
    available on PATH.
    """
    local_path.mkdir(parents=True, exist_ok=True)

    # First try modern gcloud storage
    try:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                "-r",
                gcs_path,
                str(local_path),
            ],
            check=True,
        )
        return
    except FileNotFoundError:
        # gcloud not installed; fall back to gsutil
        pass

    # Fallback: gsutil
    try:
        subprocess.run(
            [
                "gsutil",
                "-m",
                "rsync",
                "-r",
                gcs_path,
                str(local_path),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Neither `gcloud` nor `gsutil` is available on PATH. "
            "Install the Google Cloud CLI to download HELM data."
        ) from exc


def download_helm_project(
    project: str,
    local_benchmark_output_dir: Union[str, Path],
    release: Optional[str] = None,
    suite: Optional[str] = None,
) -> None:
    """
    Download HELM / HELM Lite raw results from the public GCS bucket.

    This is a thin wrapper around the commands recommended in the HELM
    documentation, e.g.:

        GCS_BENCHMARK_OUTPUT_PATH=gs://crfm-helm-public/lite/benchmark_output
        gcloud storage rsync -r \\
            ${GCS_BENCHMARK_OUTPUT_PATH}/releases/v1.10.0-canary \\
            /path/to/local/releases/v1.10.0-canary
        gcloud storage rsync -r \\
            ${GCS_BENCHMARK_OUTPUT_PATH}/runs/v1.10.0-canary \\
            /path/to/local/runs/v1.10.0-canary

    Arguments
    ---------
    project:
        HELM project name, e.g. "lite", "mmlu", "heim", "med_qa", etc.
        This becomes the {project} in:
            gs://crfm-helm-public/{project}/benchmark_output

    local_benchmark_output_dir:
        Local directory under which we create `releases/` and/or `runs/`.

    release:
        Optional release version (for HELM Lite you may want something like
        "v1.10.0-canary"). If provided, we download:
            gs://crfm-helm-public/{project}/benchmark_output/releases/{release}
        into:
            {local_benchmark_output_dir}/releases/{release}

    suite:
        Optional runs suite. For HELM Lite, the suite is usually the same as
        the release version (e.g. "v1.10.0-canary"). For classic HELM, suites
        can be names like "mmlu", "my-suite", etc. If provided, we download:
            gs://crfm-helm-public/{project}/benchmark_output/runs/{suite}
        into:
            {local_benchmark_output_dir}/runs/{suite}
    """
    local_root = Path(local_benchmark_output_dir).expanduser().resolve()
    bench_root = local_root

    base = f"gs://crfm-helm-public/{project}/benchmark_output"

    if release is not None:
        gcs_release = f"{base}/releases/{release}"
        local_release = bench_root / "releases" / release
        _gcs_rsync(gcs_release, local_release)

    if suite is not None:
        gcs_runs = f"{base}/runs/{suite}"
        local_runs = bench_root / "runs" / suite
        _gcs_rsync(gcs_runs, local_runs)


if __name__ == "__main__":
    download_helm_project(
            project="mmlu",
            local_benchmark_output_dir="data/mmlu",
            suite="mmlu",
    )
