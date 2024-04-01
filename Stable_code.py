import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablecode-completion-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablecode-completion-alpha-3b")
model.half().cuda()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids


prompt = f"What would a script that tests a GCP environment for CIS secure configs for FedRAMP Rev5 compliance look like?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.7,
  do_sample=True,
  stopping_criteria=StoppingCriteriaList([StopOnTokens()])
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))