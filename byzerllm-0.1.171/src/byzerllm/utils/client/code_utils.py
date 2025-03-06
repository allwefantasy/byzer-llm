import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from hashlib import md5
from typing import Callable, Dict, List, Optional, Tuple, Union
try:
    import docker
except ImportError:
    docker = None



# Regular expression for finding a code block
DEFAULT_MODEL="chat"
CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extensions")
UNKNOWN = "unknown"
TIMEOUT_MSG = "Timeout"
DEFAULT_TIMEOUT = 600
WIN32 = sys.platform == "win32"
PATH_SEPARATOR = WIN32 and "\\" or "/"

logger = logging.getLogger(__name__)


def get_value_from_llm_str(v:str,k:str, default_value)->Union[str,int,float,bool,None]:
    responses = extract_code(v)
    value = default_value
    for lang,code in responses:
        if lang == "json":
            try:
                value = json.loads(code)[k]
            except Exception as inst:
                pass 
    return value

def content_str(content: Union[str, List, None]) -> str:
    """Converts `content` into a string format.

    This function processes content that may be a string, a list of mixed text and image URLs, or None,
    and converts it into a string. Text is directly appended to the result string, while image URLs are
    represented by a placeholder image token. If the content is None, an empty string is returned.

    Args:
        - content (Union[str, List, None]): The content to be processed. Can be a string, a list of dictionaries
                                      representing text and image URLs, or None.

    Returns:
        str: A string representation of the input content. Image URLs are replaced with an image token.

    Note:
    - The function expects each dictionary in the list to have a "type" key that is either "text" or "image_url".
      For "text" type, the "text" key's value is appended to the result. For "image_url", an image token is appended.
    - This function is useful for handling content that may include both text and image references, especially
      in contexts where images need to be represented as placeholders.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"content must be None, str, or list, but got {type(content)}")

    rst = ""
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("Wrong content format: every element should be dict if the content is a list.")
        assert "type" in item, "Wrong content format. Missing 'type' key in content's dict."
        if item["type"] == "text":
            rst += item["text"]
        elif item["type"] == "image_url":
            rst += "<image>"
        else:
            raise ValueError(f"Wrong content format: unknown type {item['type']} within the content")
    return rst

def infer_lang(code):
    """infer the language for the code.
    TODO: make it robust.
    """
    if code.startswith("python ") or code.startswith("pip") or code.startswith("python3 "):
        return "sh"

    # check if code is a valid python code
    try:
        compile(code, "test", "exec")
        return "python"
    except SyntaxError:
        # not a valid python code
        return UNKNOWN


def check_target_codes_exists(codes: List[Tuple[str, str]], langs: List[str]) -> bool:
    """Check if there is code in a specific language in the code list.

    Args:
        codes (list): The list of code blocks.
        langs (list): The language to check.

    Returns:
        bool: True if there is code in the specified language; False otherwise.
    """
    for l, _ in codes:
        if l in langs:
            return True
    return False

def get_target_codes(codes: List[Tuple[str, str]], langs: List[str]) -> List[str]:
    """Get code in a specific language from the code list.

    Args:
        codes (list): The list of code blocks.
        langs (list): The language to check.

    Returns:
        str: The code in the specified language.
    """
    target_codes = []
    for l, code in codes:
        if l in langs:
            target_codes.append(code)
    return target_codes


def extract_code(
    text: Union[str, List], pattern: str = CODE_BLOCK_PATTERN, detect_single_line_code: bool = False
) -> List[Tuple[str, str]]:
    """Extract code from a text.

    Args:
        text (str or List): The content to extract code from. The content can be
            a string or a list, as returned by standard GPT or multimodal GPT.
        pattern (str, optional): The regular expression pattern for finding the
            code block. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable the new feature for
            extracting single line code. Defaults to False.

    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is code block but the language is not specified, the language would be "".
    """
    text = content_str(text)
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [(UNKNOWN, text)]

    # Extract both multi-line and single-line code block, separated by the | operator
    # `([^`]+)`: Matches inline code.
    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    # Extract the individual code blocks and languages from the matched groups
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    return extracted


def _cmd(lang):
    if lang.startswith("python") or lang in ["bash", "sh", "powershell"]:
        return lang
    if lang in ["shell"]:
        return "sh"
    if lang in ["ps1"]:
        return "powershell"
    raise NotImplementedError(f"{lang} not recognized in code execution")


def execute_code(
    code: Optional[str] = None,
    timeout: Optional[int] = None,
    filename: Optional[str] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[Union[List[str], str, bool]] = None,
    lang: Optional[str] = "python",
) -> Tuple[int, str, str]:
    """Execute code in a docker container.
    This function is not tested on MacOS.

    Args:
        code (Optional, str): The code to execute.
            If None, the code from the file specified by filename will be executed.
            Either code or filename must be provided.
        timeout (Optional, int): The maximum execution time in seconds.
            If None, a default timeout will be used. The default timeout is 600 seconds. On Windows, the timeout is not enforced when use_docker=False.
        filename (Optional, str): The file name to save the code or where the code is stored when `code` is None.
            If None, a file with a randomly generated name will be created.
            The randomly generated file will be deleted after execution.
            The file name must be a relative path. Relative paths are relative to the working directory.
        work_dir (Optional, str): The working directory for the code execution.
            If None, a default working directory will be used.
            The default working directory is the "extensions" directory under
            "path_to_autogen".
        use_docker (Optional, list, str or bool): The docker image to use for code execution.
            If a list or a str of image name(s) is provided, the code will be executed in a docker container
            with the first image successfully pulled.
            If None, False or empty, the code will be executed in the current environment.
            Default is None, which will be converted into an empty list when docker package is available.
            Expected behaviour:
                - If `use_docker` is explicitly set to True and the docker package is available, the code will run in a Docker container.
                - If `use_docker` is explicitly set to True but the Docker package is missing, an error will be raised.
                - If `use_docker` is not set (i.e., left default to None) and the Docker package is not available, a warning will be displayed, but the code will run natively.
            If the code is executed in the current environment,
            the code must be trusted.
        lang (Optional, str): The language of the code. Default is "python".

    Returns:
        int: 0 if the code executes successfully.
        str: The error message if the code fails to execute; the stdout otherwise.
        image: The docker image name after container run when docker is used.
    """    
    if all((code is None, filename is None)):
        error_msg = f"Either {code=} or {filename=} must be provided."
        logger.error(error_msg)
        raise AssertionError(error_msg)

    # Warn if use_docker was unspecified (or None), and cannot be provided (the default).
    # In this case the current behavior is to fall back to run natively, but this behavior
    # is subject to change.
    if use_docker is None:
        if docker is None:
            use_docker = False
            logger.warning(
                "execute_code was called without specifying a value for use_docker. Since the python docker package is not available, code will be run natively. Note: this fallback behavior is subject to change"
            )
        else:
            # Default to true
            use_docker = True

    timeout = timeout or DEFAULT_TIMEOUT
    original_filename = filename
    if WIN32 and lang in ["sh", "shell"] and (not use_docker):
        lang = "ps1"
    if filename is None:
        code_hash = md5(code.encode()).hexdigest()
        # create a file with a automatically generated name
        filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
    if work_dir is None:
        work_dir = WORKING_DIR
    filepath = os.path.join(work_dir, filename)
    file_dir = os.path.dirname(filepath)
    os.makedirs(file_dir, exist_ok=True)
    if code is not None:
        with open(filepath, "w", encoding="utf-8") as fout:
            fout.write(code)
    # check if already running in a docker container
    in_docker_container = os.path.exists("/.dockerenv")
    if not use_docker or in_docker_container:
        # already running in a docker container
        cmd = [
            sys.executable if lang.startswith("python") else _cmd(lang),
            f".\\{filename}" if WIN32 else filename,
        ]
        if WIN32:
            logger.warning("SIGALRM is not supported on Windows. No timeout will be enforced.")
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    subprocess.run,
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                try:
                    result = future.result(timeout=timeout)
                except TimeoutError:
                    if original_filename is None:
                        os.remove(filepath)
                    return 1, TIMEOUT_MSG, None
        if original_filename is None:
            os.remove(filepath)
        if result.returncode:
            logs = result.stderr
            if original_filename is None:
                abs_path = str(pathlib.Path(filepath).absolute())
                logs = logs.replace(str(abs_path), "").replace(filename, "")
            else:
                abs_path = str(pathlib.Path(work_dir).absolute()) + PATH_SEPARATOR
                logs = logs.replace(str(abs_path), "")
        else:
            logs = result.stdout
        return result.returncode, logs, None

    # create a docker client
    client = docker.from_env()
    image_list = (
        ["python:3-alpine", "python:3", "python:3-windowsservercore"]
        if use_docker is True
        else [use_docker]
        if isinstance(use_docker, str)
        else use_docker
    )
    for image in image_list:
        # check if the image exists
        try:
            client.images.get(image)
            break
        except docker.errors.ImageNotFound:
            # pull the image
            print("Pulling image", image)
            try:
                client.images.pull(image)
                break
            except docker.errors.DockerException:
                print("Failed to pull image", image)
    # get a randomized str based on current time to wrap the exit code
    exit_code_str = f"exitcode{time.time()}"
    abs_path = pathlib.Path(work_dir).absolute()
    cmd = [
        "sh",
        "-c",
        f"{_cmd(lang)} {filename}; exit_code=$?; echo -n {exit_code_str}; echo -n $exit_code; echo {exit_code_str}",
    ]
    # create a docker container
    container = client.containers.run(
        image,
        command=cmd,
        working_dir="/workspace",
        detach=True,
        # get absolute path to the working directory
        volumes={abs_path: {"bind": "/workspace", "mode": "rw"}},
    )
    start_time = time.time()
    while container.status != "exited" and time.time() - start_time < timeout:
        # Reload the container object
        container.reload()
    if container.status != "exited":
        container.stop()
        container.remove()
        if original_filename is None:
            os.remove(filepath)
        return 1, TIMEOUT_MSG, image
    # get the container logs
    logs = container.logs().decode("utf-8").rstrip()
    # commit the image
    tag = filename.replace("/", "")
    container.commit(repository="python", tag=tag)
    # remove the container
    container.remove()
    # check if the code executed successfully
    exit_code = container.attrs["State"]["ExitCode"]
    if exit_code == 0:
        # extract the exit code from the logs
        pattern = re.compile(f"{exit_code_str}(\\d+){exit_code_str}")
        match = pattern.search(logs)
        exit_code = 1 if match is None else int(match.group(1))
        # remove the exit code from the logs
        logs = logs if match is None else pattern.sub("", logs)

    if original_filename is None:
        os.remove(filepath)
    if exit_code:
        logs = logs.replace(f"/workspace/{filename if original_filename is None else ''}", "")
    # return the exit code, logs and image
    return exit_code, logs, f"python:{tag}"


def _remove_check(response):
    """Remove the check function from the response."""
    # find the position of the check function
    pos = response.find("def check(")
    if pos == -1:
        return response
    return response[:pos]


def eval_function_completions(
    responses: List[str],
    definition: str,
    test: Optional[str] = None,
    entry_point: Optional[str] = None,
    assertions: Optional[Union[str, Callable[[str], Tuple[str, float]]]] = None,
    timeout: Optional[float] = 3,
    use_docker: Optional[bool] = True,
) -> Dict:
    """(openai<1) Select a response from a list of responses for the function completion task (using generated assertions), and/or evaluate if the task is successful using a gold test.

    Args:
        responses (list): The list of responses.
        definition (str): The input definition.
        test (Optional, str): The test code.
        entry_point (Optional, str): The name of the function.
        assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.
            When provided, only the responses that pass the assertions will be considered for the actual test (if provided).
        timeout (Optional, float): The timeout for executing the code.

    Returns:
        dict: The success metrics.
    """
    n = len(responses)
    if assertions is None:
        # no assertion filter
        success_list = []
        for i in range(n):
            response = _remove_check(responses[i])
            code = (
                f"{response}\n{test}\ncheck({entry_point})"
                if response.startswith("def")
                else f"{definition}{response}\n{test}\ncheck({entry_point})"
            )
            success = execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
            success_list.append(success)
        return {
            "expected_success": 1 - pow(1 - sum(success_list) / n, n),
            "success": any(s for s in success_list),
        }
    if callable(assertions) and n > 1:
        # assertion generator
        assertions, gen_cost = assertions(definition)
    else:
        assertions, gen_cost = None, 0
    if n > 1 or test is None:
        for i in range(n):
            response = responses[i] = _remove_check(responses[i])
            code = (
                f"{response}\n{assertions}" if response.startswith("def") else f"{definition}{response}\n{assertions}"
            )
            succeed_assertions = execute_code(code, timeout=timeout, use_docker=use_docker)[0] == 0
            if succeed_assertions:
                break
    else:
        # just test, no need to check assertions
        succeed_assertions = False
        i, response = 0, responses[0]
    if test is None:
        # no test code
        return {
            "index_selected": i,
            "succeed_assertions": succeed_assertions,
            "gen_cost": gen_cost,
            "assertions": assertions,
        }
    code_test = (
        f"{response}\n{test}\ncheck({entry_point})"
        if response.startswith("def")
        else f"{definition}{response}\n{test}\ncheck({entry_point})"
    )
    success = execute_code(code_test, timeout=timeout, use_docker=use_docker)[0] == 0
    return {
        "index_selected": i,
        "succeed_assertions": succeed_assertions,
        "success": success,
        "gen_cost": gen_cost,
        "assertions": assertions,
    }


_FUNC_COMPLETION_PROMPT = "# Python 3{definition}"
_FUNC_COMPLETION_STOP = ["\nclass", "\ndef", "\nif", "\nprint"]

class PassAssertionFilter:
    def __init__(self, assertions):
        self._assertions = assertions        
        self.cost = 0
        self.metrics = self.responses = None

    def pass_assertions(self, context, response, **_):        
        responses = response[0].output        
        metrics = eval_function_completions(responses, context["definition"], assertions=self._assertions)
        self._assertions = metrics["assertions"]
        self.cost += metrics["gen_cost"]
        self.metrics = metrics
        self.responses = responses
        return metrics["succeed_assertions"]

