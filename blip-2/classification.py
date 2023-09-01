from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
import torch


class BLIP2:
    def __init__(self, ):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def __call__(self, *args, **kwargs):
        return self.get_img_embedding(image=self.image)

    def get_img_embedding(self, image):
        """
        Turn a list of image inputs into tensor of embedding vectors
        images should be of shape (batch_size, channels, height, width)
        """
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                torch_dtype=torch.float16)
        self.model.to(self.device)
        
        image_tensors = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        image_tensors = image_tensors.to(self.device, torch.float16)  # type: ignore

        # pass images through the vision model and then the qformer to get query-conditional image features
        query_outputs = self.model.get_qformer_features(**image_tensors)  # tuple (last_hidden_state, pooler_output)
        return query_outputs

    def get_text_embedding(self, texts):
        """
        Turn a list of text inputs into tensor of embedding vectors.
        texts is a list of strings to embed.
        """

        text_tokens = self.model.text_tokenizer(texts, padding=True, return_tensors='pt')
        text_tokens = text_tokens.to(self.device)

        text_outputs = blip2model.get_text_features(**text_tokens, output_hidden_states=True)  # type: ignore
        # extract [CLS] embedding from last hidden state, shape (batch_size, hidden_size)
        text_features = text_outputs['hidden_states'][-1][:, 0, :]
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


if __name__ == "__main__":
    print(BLIP2()())
