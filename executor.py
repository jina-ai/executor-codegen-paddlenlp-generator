from jina import Executor, DocumentArray, requests
from paddlenlp.transformers import CodeGenTokenizer, CodeGenForCausalLM
import paddle
class CodeGenerator(Executor):
    def __init__(self,
                model_name:str="Salesforce/codegen-350M-mono",
                min_length:int=256,
                max_length:int=1024,
                decode_strategy:str='sampling',
                top_k:int=5,
                repetition_penalty:float=1.1,
                temperature:float=0.5,
                candidate_number:int=5,
                *args,
                **kwargs):
        super().__init__(*args,**kwargs)
        self.tokenizer = CodeGenTokenizer.from_pretrained(model_name)
        self.model = CodeGenForCausalLM.from_pretrained(model_name)
        self._min_length = min_length
        self._max_length = max_length
        self._decode_stragtegy = decode_strategy
        self._top_k = top_k
        self._repetition_penalty = repetition_penalty
        self._temperaturea = temperature
        self._candidate_number = candidate_number

    @requests
    def generate_code(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            prompt = doc.tags["prompt"]
            ged_code = doc.tags["code"]
            inputs = self.tokenizer([prompt+ged_code])
            inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
            outputs = {}
            for _ in range(self._candidate_number):
                output, score = self.model.generate(inputs['input_ids'],
                               min_length=self._min_length,
                               decode_strategy=self._decode_stragtegy,
                               top_k=self._top_k,
                               repetition_penalty=self._repetition_penalty,
                               temperature=self._temperaturea)
                outputs[score.item()] = self.tokenizer.decode(
                                                       output[0],
                                                       skip_special_tokens=True,
                                                       spaces_between_special_tokens=False)
            doc.tags["candidates"] = outputs
