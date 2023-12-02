from transformers import pipeline

model = pipeline("text-generation", model="gpt2")

sentence = model(
    "Hi, My name is John Cena, I am here",
    do_sample=True,
    top_k=50,
    temperature=0.9,
    max_length=100  # Use max_length to control the length of the generated text
)

for i in sentence:
    print(i["generated_text"])
