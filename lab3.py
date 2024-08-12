import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_story(prompt_text, model, tokenizer, max_length=200):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate the story
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            no_repeat_ngram_size=2, early_stopping=True,
                            pad_token_id=tokenizer.eos_token_id)

    # Decode the generated story
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_story

def main():
    st.title("Story Generator")
    st.write("provide me the promt")

    st.sidebar.title('Customization')
    model_name = st.sidebar.selectbox(
        'Choose a model',
        ['gpt2', 'distilgpt2']
    )
    max_length = st.sidebar.slider('Maximum story length:', 50, 500, value=200)

    # Load the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    user_prompt = st.text_input("Enter a prompt for the story:")
   
    if user_prompt:
        story = generate_story(user_prompt, model, tokenizer, max_length)
        st.write("Generated Story:")
        st.write(story)

if __name__ == "__main__":
    main()
