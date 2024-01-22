from typing import Union, List, Tuple

from FlagEmbedding import FlagReranker


def embed_rerank(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], extract_params={}) -> Union[
    Tuple[Tuple[str, str], float], List[Tuple[Tuple[str, str], float]]]:
    scores = self.compute_score(sentence_pairs,
                                batch_size=extract_params.get("batch_size", 256),
                                max_length=extract_params.get("max_length", 512)
                                )
    if not isinstance(scores, List):
        return (sentence_pairs, scores)
    else:
        return sorted(list(zip(sentence_pairs, scores)), key=lambda x: x[1], reverse=True)


def init_model(model_dir, infer_params, sys_conf={}):
    model = FlagReranker(model_dir, use_fp16=sys_conf.get("use_fp16", True))
    import types
    model.embed_rerank = types.MethodType(embed_rerank, model)
    return (model, None)
