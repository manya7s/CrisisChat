from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

mdlName = "microsoft/DialoGPT-medium"
tkzr = AutoTokenizer.from_pretrained(mdlName)
mdl = AutoModelForCausalLM.from_pretrained(mdlName)

if tkzr.pad_token is None:
    tkzr.pad_token = tkzr.eos_token

def chatWithHf(prmpt, chatHist=None):
    try:
        newInpIds = tkzr.encode(prmpt + tkzr.eos_token, return_tensors="pt")
        
        botInpIds = (
            newInpIds if chatHist is None
            else torch.cat([chatHist, newInpIds], dim=-1)
        )
        
        attMask = torch.ones_like(botInpIds)
        chatHist = mdl.generate(
            botInpIds,
            max_length=1000,
            pad_token_id=tkzr.pad_token_id,
            attention_mask=attMask,
        )
        
        #respones
        rsp = tkzr.decode(chatHist[:, botInpIds.shape[-1]:][0], skip_special_tokens=True)
        return rsp, chatHist
    except Exception as e:
        return f"Error: {e}", chatHist

if __name__ == "__main__":
    os.system("cls")
    print("Hello, my name is Crisis, let's talk! (type 'bye' to quit)")
    chatHist = None
    while True:
        usrInp = input("You: ")
        if usrInp.lower() == "bye":
            print("Goodbye!")
            break
        print("...")
        rply, chatHist = chatWithHf(usrInp, chatHist)
        print(f"Crisis: {rply}")
