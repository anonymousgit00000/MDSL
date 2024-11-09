import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from federatedscope.llm.model.model_builder import get_llm
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from peft import PeftModel, LoraConfig, get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from transformers import LlamaTokenizer, AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, \
    BertModel, BertTokenizer
from dataloader import generate_data_LLM_gsm, generate_data_LLM_code, generate_data_router_gsm, generate_data_router_code
from federatedscope.llm.dataset.llm_dataset import DefaultToken

BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.90
# EPISILO = 0.9
MEMORY_CAPACITY = 2000
shared_MEMORY_CAPACITY = 500
Q_NETWORK_ITERATION = 100
directory = './save' + '/'
mode = 'train'
# mode = 'test'


# NUM_ACTIONS  defined by clients

NUM_STATES = 768


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

def get_tokenizer(model_name, cache_dir, tok_len=128, pkg='huggingface_llm'):

    assert pkg in ['huggingface_llm', 'modelscope_llm'], \
        f'Not supported package {pkg}.'

    if pkg == 'huggingface_llm':
        from transformers import AutoTokenizer, LlamaTokenizer
    elif pkg == 'modelscope_llm':
        from modelscope import AutoTokenizer, LlamaTokenizer


    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # use_auth_token=token,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens = dict()

    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    #print("------------------------pretrained_embeddings-------------------------")

    #print(len(tokenizer))

    return tokenizer, num_new_tokens


class Environment():
    def __init__(self):
        self.model_name = 1

    def step(self, action_index, model, data_number, flag):

        new_vocab_size = 32001
        model.resize_token_embeddings(new_vocab_size)
        model = PeftModel.from_pretrained(model, "./federatedscope/save_pretrained_gsm_1000_0.3.0", adapter_name="adapter1")
        model.load_adapter("./federatedscope/save_pretrained_rosetta_1000_0.3.0", adapter_name="adapter2")
        
        # action_list = [] defined by clients

        weight = action_list[action_index]

        model.add_weighted_adapter(
            adapters=["adapter1", "adapter2"],
            weights = weight,
            # weights=[0.9, 0.1],
            adapter_name="weighted_lora",
            combination_type="cat"
        )

        model.set_adapter("weighted_lora")

        if flag == True:
            
            input_ids, labels, attention_mask = generate_data_LLM_gsm(data_number) 
        else:

            input_ids, labels, attention_mask = generate_data_LLM_code(data_number) 
        
        outputs = model(input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask)
        loss = outputs.loss.detach()
        ppl = np.exp(loss.cpu())


        if flag == True:
            reward = -loss * 100  
        else:
            reward = -loss * 100  

        next_state = 0  
        model.delete_adapter("weighted_lora")

        return next_state, reward, ppl


env = Environment()  # initilize env

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        # self.eval_net, self.target_net = Net(), Net()
        self.eval_net = Net()

        self.learn_step_counter = 0
        self.math_memory_counter = 0
        self.math_shared_memory_counter = 0
        self.code_memory_counter = 0
        self.code_shared_memory_counter = 0

        self.math_memory = np.zeros((MEMORY_CAPACITY, NUM_STATES + 3)) 
        self.math_shared_memory = np.zeros((shared_MEMORY_CAPACITY, NUM_STATES + 3))  
        self.code_memory = np.zeros((MEMORY_CAPACITY, NUM_STATES + 3))  
        self.code_shared_memory = np.zeros((shared_MEMORY_CAPACITY, NUM_STATES + 3)) 

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, weight_decay=0.01)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPISILO):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) 
        if np.random.rand() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] 
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS-1)
            action = action 
        return action


    def store_math_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.math_memory_counter 
        self.math_memory[index, :] = transition
        self.math_memory_counter += 1
    
    def store_code_shared_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        # print("=======================")
        # print("transition =", transition)
        index = self.code_shared_memory_counter 
        self.code_shared_memory[index, :] = transition
        self.code_shared_memory_counter += 1


    def store_code_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.code_memory_counter % MEMORY_CAPACITY
        self.code_memory[index, :] = transition
        self.code_memory_counter += 1
    
    def store_math_shared_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.math_shared_memory_counter % MEMORY_CAPACITY
        self.math_shared_memory[index, :] = transition
        self.math_shared_memory_counter += 1

    def learn(self, flag):

        if flag == True:
            math_sample_index = np.random.choice(MEMORY_CAPACITY, int(BATCH_SIZE/2))
            code_shared_sample_index = np.random.choice(shared_MEMORY_CAPACITY, int(BATCH_SIZE/2))
            math_batch_memory = self.math_memory[math_sample_index, :]
            code_shared_batch_memory = self.code_shared_memory[code_shared_sample_index, :]
            batch_memory = np.concatenate((math_batch_memory, code_shared_batch_memory))
            
            batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
            batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
            batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
            batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])
        else:
            code_sample_index = np.random.choice(MEMORY_CAPACITY, int(BATCH_SIZE/2))
            math_shared_sample_index = np.random.choice(shared_MEMORY_CAPACITY, int(BATCH_SIZE/2))
            code_batch_memory = self.code_memory[code_sample_index, :]
            math_shared_batch_memory = self.math_memory[math_shared_sample_index, :]
            batch_memory = np.concatenate((code_batch_memory, math_shared_batch_memory))
            
            batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
            batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
            batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
            batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])


        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_target = batch_reward 
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

    def save(self):
        torch.save(self.eval_net.state_dict(), directory + 'DQN_fed_cat.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.eval_net.load_state_dict(torch.load(directory + 'DQN_fed_cat_fix.pth'))
        # self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def reverse_flag(flag):
    return not flag

def release_memory(model):
    del model  
    torch.cuda.empty_cache()  

def main():
    dqn = DQN()
    episodes = 1600
    EPISILO = 0.7
    shared_math_count = 0
    shared_code_count = 0
    input_ids_gsm, attention_mask_gsm = generate_data_router_gsm()  # input_ids is corpus, containing 1000 sentences

    input_ids_code, attention_mask_code = generate_data_router_code()  # input_ids is corpus, containing 1000 sentences

    bert = BertModel.from_pretrained('bert-base-uncased')
    LLM_model_name = "baffo32/decapoda-research-llama-7B-hf"
    LLM_model = AutoModelForCausalLM.from_pretrained(LLM_model_name, device_map="auto")
    print("Collecting Experience....")
    flag = True  # 1 equals to math, 0 equals to code
    total_data_number = 100
    # total_data_number = 10
    data_number = -1
    # data_number = 0
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    total_reward = 0
    total_cosine_similarity = 0
    total_pca_similarity = 0
    cosine_similarity_list = []
    pca_cosine_similarity_list = []


    if mode == 'train':
        for i in range(episodes):
            print("=========================")
            print("episode =", i)

            if i != 0 and i % (2 * total_data_number) == 0:
               EPISILO = EPISILO + 0.1 

            if i != 0 and i % total_data_number == 0:
                flag = reverse_flag(flag)
                print("==========reverse flag!===============")


                data_number = 0
                total_reward = 0
            else:
                data_number += 1
                # data_number = 0

            if flag == True:
                input_ids_bert = input_ids_gsm[data_number].unsqueeze(0)
                attention_mask_bert = attention_mask_gsm[data_number].unsqueeze(0)
            else:
                input_ids_bert = input_ids_code[data_number].unsqueeze(0)
                attention_mask_bert = attention_mask_code[data_number].unsqueeze(0)
            with torch.no_grad():
                outputs_bert = bert(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
            state = outputs_bert.pooler_output
            state = state[0].cpu()
            action = dqn.choose_action(state, EPISILO)

            next_state, reward, ppl = env.step(action, LLM_model, data_number, flag)

            reward = reward.cpu()
            total_reward += reward

            if flag == True:
                
                dqn.store_math_transition(state, action, reward, next_state)
                if shared_math_count < shared_MEMORY_CAPACITY:
                    dqn.store_math_shared_transition(state, action, reward, next_state)
                    shared_math_count += 1
            else:
                dqn.store_code_transition(state, action, reward, next_state)
                if shared_code_count < shared_MEMORY_CAPACITY:
                    dqn.store_code_shared_transition(state, action, reward, next_state)
                    shared_code_count += 1


            if flag == True:
                if dqn.math_memory_counter >= BATCH_SIZE/2 and dqn.code_shared_memory_counter >= BATCH_SIZE/2:
                    print("==================")
                    print("start to learn!")
                    dqn.learn(flag)
            else:
                if dqn.code_memory_counter >= BATCH_SIZE/2 and dqn.math_shared_memory_counter >= BATCH_SIZE/2:
                    print("==================")
                    print("start to learn!")
                    dqn.learn(flag)
        
            if (i+1) % total_data_number == 0:       
                print("episode: {} , the total_reward is {}".format(i, total_reward))
            if i == episodes - 1:
                dqn.save()
            

if __name__ == '__main__':
    main()
