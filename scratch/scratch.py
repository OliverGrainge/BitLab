import torch
import transformers as tf

model_name = "SpectraSuite/TriLM_99M_Unpacked"

# Please adjust the temperature, repetition penalty, top_k, top_p and other sampling parameters according to your needs.
pipeline = tf.pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"dtype": torch.float16},
    device_map="auto",
)

# These are base (pretrained) LLMs that are not instruction and chat tuned. You may need to adjust your prompt accordingly.
print(pipeline("Once upon a time"))
