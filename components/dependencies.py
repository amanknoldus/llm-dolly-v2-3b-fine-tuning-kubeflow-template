from typing import List

DEPENDENCY_VERSIONS = {
    "numpy": "numpy==1.24.4",
    "pandas": "pandas==2.0.2",
    "fsspec": "fsspec==2023.6.0",
    "pyarrow": "pyarrow==12.0.1",
    "gcsfs": "gcsfs==2023.6.0",
    "google-cloud-storage": "google-cloud-storage==2.10.0",
    "google-cloud-aiplatform": "google-cloud-aiplatform==1.29.0",
    "google-cloud-build": "google-cloud-build==3.20.0",
    "google": "google~=3.0.0",
    "psutil": "psutil==5.9.5",
    "datasets": "datasets>=2.10.0,<3",
    "transformers": "transformers[torch]>=4.28.1,<5",
    "torch": "torch>=1.13.1,<2",
    "accelerate": "accelerate>=0.16.0,<1",

}


def resolve_dependencies(*names) -> List[str]:
    return [DEPENDENCY_VERSIONS[name] for name in names]
