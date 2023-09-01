from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


class BLIP2:
    def __init__(self, ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def __call__(self, *args, **kwargs):
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                              torch_dtype=torch.float16)
        model.to(self.device)
        prompt = "Question: how many cats are there? Answer:"
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


if __name__ == "__main__":
    print(BLIP2()())
