from ipex_llm.transformers import AutoModelForCausalLM

model_path = 'D:\\LLM\\LLM_DEMO\\ModelCache\\Baichuan2-7B-chat'

# model_in_4bit = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
#                                                      load_in_4bit=True,
#                                                      trust_remote_code=True,
#                                                      #use_cache=True,
#                                                      #model_hub='modelscope',
#                                                      )
# model_in_4bit_gpu = model_in_4bit.to('xpu')


save_directory = 'D:\\LLM\\LLM_DEMO\\ModelCache\\baichuan2-7b-chat-ipex-llm-INT4'

model_in_4bit = AutoModelForCausalLM.load_low_bit(save_directory, trust_remote_code=True)
#model_in_4bit_gpu = model_in_4bit.to('xpu')



from transformers import AutoTokenizer

from transformers import GenerationConfig


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                          trust_remote_code=True,
                                          )

model_in_4bit.generation_config = GenerationConfig.from_pretrained(model_path)



chat_history = []

print('-'*20, 'Stream Chat', '-'*20, end="\n")
while True:
    prompt = input("Input: ")
    if prompt.strip() == "stop": # let's stop the conversation when user input "stop"
        print("Stream Chat with Baichuan 2 (7B) stopped.")
        break
    chat_history.append({"role": "user", "content": prompt})
    position = 0
    for response in model_in_4bit.chat(tokenizer, chat_history, stream=True):
        print(response[position:], end='', flush=True)
        position = len(response)
    print()
    chat_history.append({"role": "assistant", "content": response})


# import torch

# with torch.inference_mode():
#     prompt = 'Q: What is CPU?\nA:'
    
#     # tokenize the input prompt from string to token ids;
#     # with .to('xpu') specifically for inference on Intel GPUs
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')

#     # predict the next tokens (maximum 32) based on the input token ids
#     output = model_in_4bit_gpu.generate(input_ids,
#                             max_new_tokens=32)

#     # decode the predicted token ids to output string
#     output = output.cpu()
#     output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     print('-'*20, 'Output', '-'*20)
#     print(output_str)