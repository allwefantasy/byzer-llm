from typing import List, Optional, Dict, Any
import os
import subprocess
import re
from loguru import logger
from byzerllm.apps.byzer_storage import env


def _check_java_version(java_home: str):
    try:
        output = subprocess.check_output(
            [f"{java_home}/bin/java", "-version"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        version_line = output.splitlines()[0]
        version_match = re.search(r'version "(\d+)', version_line)
        if version_match:
            version = version_match.group(1)
            version_parts = version.split(".")
            major_version = int(version_parts[0])
            print(major_version)
            if major_version < 21:
                raise ValueError(
                    f"Java version {version} is not supported. JDK 21 or higher is required."
                )
        else:
            raise ValueError("Could not determine Java version.")
    except (subprocess.CalledProcessError, ValueError) as e:
        raise ValueError(f"Error checking Java version: {str(e)}")


def connect_cluster(
    address: str = "auto",
    java_home: Optional[str] = None,
    code_search_path: Optional[List[str]] = None,
    init_options: Optional[Dict[str, Any]] = {},
):
    import ray

    job_config = None
    env_vars = {}

    java_home = java_home if java_home else os.environ.get("JAVA_HOME")

    v = env.detect_env()
    if v.java_home and v.java_version == "21":
        logger.info(f"JDK 21 will be used ({v.java_home})...")
        java_home = v.java_home

    if java_home:
        path = os.environ.get("PATH")
        env_vars = {
            "JAVA_HOME": java_home,
            "PATH": f"""{os.path.join(java_home,"bin")}:{path}""",
        }
        if code_search_path:
            if java_home:
                _check_java_version(java_home)
            job_config = ray.job_config.JobConfig(
                code_search_path=code_search_path, runtime_env={"env_vars": env_vars}
            )
    if not java_home and code_search_path:
        logger.warning("code_search_path is ignored because JAVA_HOME is not set")

    init_options = {**{"log_to_driver": False}, **init_options}
    ray.init(
        address=address,
        namespace="default",
        ignore_reinit_error=True,
        job_config=job_config,
        **init_options,
    )
    return env_vars
