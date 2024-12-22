from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Choosing a model:
#For this example, I'll be using facebook/blenderbot-400M-distill 
#because it has an open-source license and runs relatively fast.
model_name = "facebook/blenderbot-400M-distill"

#Fetch the model and initialize a tokenizer
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#chatbot will also reference the previous conversations when generating output
#So use list to store the conversation history
conversation_history = []


"""


#I will pass my history chat to the model ALONG with my input
#each element separated by the newline character '\n'
history_string = "\n".join(conversation_history)

#Fetch prompt from user
input_text ="hello, how are you doing?"

#Tokenization of user prompt and chat history
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

#Result:
#{'input_ids': tensor([[1710,   86,   19,  544,  366,  304,  929,   38]]), 
#'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}


#This attribute provides a mapping of pretrained models
tokenizer.pretrained_vocab_files_map


#Generate output (as tokens) from the model
#Now input is tokenized, I will past this input into the model and then generate the response
outputs = model.generate(**inputs)
print(outputs)
#and run the file 
#->tensor([[   1,  281,  476,  929,  731,   21,  281,  632,  929,  712,  731,   21,
#          855,  366,  304,   38,  946,  304,  360,  463, 5459, 7930,   38,    2]])


#The ouput is a dictionary and contain tokens, now need to be decoded
#decode the first index of outputs to see the response in plaintext.
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)

#->I'm doing well. I am doing very well. How are you? Do you have any hobbies?



#add both the input and response to conversation_history in plaintext.
conversation_history.append(input_text)
conversation_history.append(response)
print(conversation_history)

#-> ['hello, how are you doing?', "I'm doing well. I am doing very well. How are you? Do you have any hobbies?"]


"""







#LOOP
#Put everything in a loop and run a whole conversation!
while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)













