from transformers import AutoTokenizer, AutoModelForCausalLM

llm_model = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(llm_model)

LLM = AutoModelForCausalLM.from_pretrained(llm_model)

while True:

    prompt = input("Enter Question : ")

    if prompt.lower() == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt")

    output_ids = LLM.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=250,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print("\nAI :", response)
    print()
