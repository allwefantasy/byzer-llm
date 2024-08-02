from typing import List, Dict, Any
import byzerllm
from loguru import logger
from pydantic import BaseModel
from byzerllm.apps.utils import TagExtractor, Tag
import importlib
import json


def read_alpaca_zh():
    with importlib.resources.path(
        "byzerllm.apps.byzer_storage", "alpaca_zh.json"
    ) as json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)


class QAPair(BaseModel):
    question: str
    answer: str


@byzerllm.prompt()
def _convert_pretrain_text_to_instruction(text: str, num: int = 10) -> str:
    """
    根据提供的信息，生成多个相关的问题，这些问题的回答，最终要能覆盖里面所有的信息,生成 {{num}} 个。

    下面是一些信息：

    {{text}}

    现在请开始生成问题，请将每个问题使用<_question_></_question_>标签包裹，每个回答使用<_answer_></_answer_>标签包裹，最后
    每个一组问题和回答使用<_group_></_group_>标签包裹。
    """


@byzerllm.prompt()
def get_more(text: str, questions_and_answers: str, num: int = 10) -> str:
    """
    下面是一些问答信息：

    {{text}}

    下面是已有的问题和回答：

    {{questions_and_answers}}

    下面是已经生成的问题和回答，请继续生成{{num}}个问题和回答，不要和前面的重复，请将每个问题使用<_question_></_question_>标签包裹，每个回答使用<_answer_></_answer_>标签包裹，最后
    每个一组问题和回答使用<_group_></_group_>标签包裹。
    """


def to_qa_pairs(text: str, llm, num: int = 30) -> List[QAPair]:
    logger.info(f"Starting to generate QA pairs for text: {text[:100]}...")

    logger.info("Generating initial QA pairs...")
    v = _convert_pretrain_text_to_instruction.with_llm(llm).run(text)
    logger.info(f"Initial QA pairs generated. Length: {len(v)}")

    max_turns = int(num / 10)
    if max_turns < 1:
        max_turns = 1
    logger.info(f"Will generate additional QA pairs for {max_turns} turns")

    turn = 0
    while turn < max_turns:
        turn += 1
        logger.info(f"Generating additional QA pairs (turn {turn}/{max_turns})...")
        t = get_more.with_llm(llm).run(text, v)
        v += t
        logger.info(f"Additional QA pairs generated. New total length: {len(v)}")

    logger.info(f"Extracting QA pairs from generated text...{v}")
    root_tag = TagExtractor(v).extract()
    qa_pairs = []
    for i, item in enumerate(root_tag.content):
        qas = item.content
        if len(qas) == 2:
            if (
                qas[0].start_tag == "<_question_>"
                and qas[0].end_tag == "</_question_>"
                and qas[1].start_tag == "<_answer_>"
                and qas[1].end_tag == "</_answer_>"
            ):
                qa_pairs.append(QAPair(question=qas[0].content, answer=qas[1].content))
                logger.debug(
                    f"Extracted QA pair {i+1}: Q: {qas[0].content[:50]}... A: {qas[1].content[:50]}..."
                )
            else:
                logger.warning(
                    f"Skipping item {i+1} due to incorrect tags: {qas[0].start_tag}, {qas[0].end_tag}, {qas[1].start_tag}, {qas[1].end_tag}"
                )
        else:
            logger.warning(
                f"Skipping item {i+1} due to incorrect number of elements: {len(qas)}"
            )

    logger.info(f"Extracted {len(qa_pairs)} valid QA pairs.")
    return qa_pairs
