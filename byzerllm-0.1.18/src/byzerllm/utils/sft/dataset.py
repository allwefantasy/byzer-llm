import json
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length,**kwargs):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if self.bos_token_id is None and self.eos_token_id is None:
            print("bos_token_id or eos_token_id are both unset",flush=True)

        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.max_seq_length = max_seq_length
        print('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        print("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        self.system_msg = kwargs.get('system_msg', 'You are a helpful assistant. Think it over and answer the user question correctly.\n')
        self.user_role = kwargs.get('user_role', 'User')
        self.assistant_role = kwargs.get('assistant_role', 'Assistant')
        self.user_role_prefix = f"{self.user_role}:"
        self.assistant_role_prefix = f"{self.assistant_role}:"

        self.dataset_tokens_count = 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
                
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        # 收集多轮对话
        utterances = [self.system_msg]
        for x in conversation:
            utterances.append(f"{self.user_role_prefix}{x['human']}\n")
            utterances.append(f"{self.assistant_role_prefix}:{x['assistant']}\n")
        utterances_ids = self.tokenizer(utterances).input_ids

        # 模型的输入格式为：<s>User:input1</s>target1</s>input2</s>target2</s>...
        if self.bos_token_id is None:
            input_ids = []
            target_mask = []  # 用于对input进行mask，只计算target部分的loss           
        else:
            input_ids = [self.bos_token_id]
            target_mask = [0]  # 用于对input进行mask，只计算target部分的loss

        if self.eos_token_id is None:
            input_ids_suffix   = []
            target_mask_suffix = 0
        else:
            input_ids_suffix   = [self.eos_token_id]
            target_mask_suffix = 1
        
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + input_ids_suffix)
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + target_mask_suffix)
            else:
                target_mask += [1] * (len(utterances_id) + target_mask_suffix)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        self.dataset_tokens_count += len(input_ids)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
