from byzerllm.apps.utils import TagExtractor

# Example usage:

sample_text = """<_group_> <_question_>祝海林的生日是哪一天？</_question_> <_answer_>祝海林的生日是2月1号。</_answer_> </_group_>
<_group_> <_question_>祝海林的生日在哪个季节？</_question_> <_answer_>祝海林的生日在冬季，因为2月是冬季的一部分。</_answer_> </_group_>
<_group_> <_question_>祝海林的生日在2月的哪一天？</_question_> <_answer_>祝海林的生日在2月的第一天，即2月1号。</_answer_> </_group_>
<_group_> <_question_>祝海林的生日是否在中国的春节期间？</_question_> <_answer_>这取决于具体年份的春节日期，但通常2月1号可能接近或就在春节期间。</_answer_> </_group_>
<_group_> <_question_>祝海林的生日是否在公历的2月？</_question_> <_answer_>是的，祝海林的生日在公历的2月1号。</_answer_> </_group_>"""

# Parse as list of dictionaries
result_list = TagExtractor(sample_text).extract()
print(result_list)
for item in result_list.content:
    item.parent = None
    print(item.start_tag,item.content)
    for item1 in item.content:
        print("=="+item1.start_tag)
        item1.parent = None
        print(item1.model_dump_json(indent=2)) 